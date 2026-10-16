"""The scheduled drift check: score the latest window with each detector and store alerts.

A detector is the SAE or the embedding representation plus its baseline and calibrated threshold.
Both turn the window's requests into one ID list per prompt (top-k SAE features, or a one-item
cluster list), which the per-ID ranking of what drifted (feature_drift) uses to explain an alert.
The SAE detector scores a window with the Jensen-Shannon divergence of those lists (score_window);
the embedding detector scores it with the mean-embedding distance (monitor/embed_mean.py) and keeps
its 50 clusters only to explain alerts. The window is the latest window_size requests, because
each threshold is only valid for the window size it was calibrated with.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import numpy as np

from backend import storage
from monitor.baseline import baseline_path, load_baseline
from monitor.calibrate import load_threshold, threshold_path
from monitor.config import load_config
from monitor.drift import FeatureDrift, feature_drift, score_window
from monitor.embed_drift import assign_clusters, embed_baseline_path, embed_prompts, load_centroids
from monitor.embed_mean import load_mean_baseline, mean_baseline_path, mean_distance
from monitor.labels import get_label

TOP_N = 5
EXAMPLES_PER_CLUSTER = 2
EXAMPLE_CHARS = 120

# Each request in a window is a dict with "id", "prompt" and "features" (top-k feature IDs).
ToSets = Callable[[Sequence[dict]], list[list[int]]]
Describe = Callable[[int, Sequence[dict], Sequence[Sequence[int]]], str]
Score = Callable[[Sequence[dict], Sequence[Sequence[int]]], float]


@dataclass(frozen=True)
class Detector:
    """One representation of a window, with the baseline and threshold it is judged against."""

    name: str
    baseline_counts: dict[int, int]
    baseline_n: int
    threshold: float
    to_sets: ToSets
    describe: Describe
    # How a window is scored; None means the Jensen-Shannon divergence of its ID lists.
    score: Score | None = None

    def score_window(self, requests: Sequence[dict], sets: Sequence[Sequence[int]]) -> float:
        """The window's drift score, compared against the threshold."""
        if self.score is None:
            return score_window(self.baseline_counts, sets)
        return self.score(requests, sets)


def rank_drift(detector: Detector, sets: Sequence[Sequence[int]]) -> list[FeatureDrift]:
    """Run the per-ID drift test on a window's ID lists, most significant change first."""
    window_counts = Counter(chain.from_iterable(sets))
    return feature_drift(detector.baseline_counts, detector.baseline_n, window_counts, len(sets))


def describe_drift(
    detector: Detector,
    drifts: Sequence[FeatureDrift],
    requests: Sequence[dict],
    sets: Sequence[Sequence[int]],
) -> list[dict]:
    """Turn drift results into JSON-ready dicts with each ID's description."""
    return [
        {
            "id": d.feature_id,
            "label": detector.describe(d.feature_id, requests, sets),
            "baseline_rate": d.baseline_rate,
            "window_rate": d.window_rate,
            "ratio": d.ratio,
            "adjusted_p": d.adjusted_p,
        }
        for d in drifts
    ]


def top_drifting(
    detector: Detector, requests: Sequence[dict], sets: Sequence[Sequence[int]], n: int = TOP_N
) -> list[dict]:
    """Rank the window's IDs with the per-ID drift test and describe the top n."""
    return describe_drift(detector, rank_drift(detector, sets)[:n], requests, sets)


class DriftJob:
    """Checks the latest window each time it runs; skips when no new request has arrived."""

    def __init__(
        self, detectors: Sequence[Detector], window_size: int, path: Path | None = None
    ) -> None:
        self.detectors = list(detectors)
        self.window_size = window_size
        self.path = path
        self.last_checked_id: int | None = None

    def latest_window(self) -> list[dict]:
        """Return the latest window_size requests, oldest first, with their feature IDs."""
        requests = storage.recent_requests(self.window_size, self.path)[::-1]
        hits = storage.hits_for_requests([r["id"] for r in requests], self.path)
        return [
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "features": [h["feature_id"] for h in hits[r["id"]]],
            }
            for r in requests
        ]

    def run(self) -> list[int]:
        """Score the latest full window with every detector, store each score, return new alerts.

        Every score is stored, alert or not, so a quiet run can be told apart from no checks.
        """
        requests = self.latest_window()
        if len(requests) < self.window_size or requests[-1]["id"] == self.last_checked_id:
            return []
        self.last_checked_id = requests[-1]["id"]

        first_id, last_id = requests[0]["id"], requests[-1]["id"]
        alert_ids = []
        for detector in self.detectors:
            sets = detector.to_sets(requests)
            score = detector.score_window(requests, sets)
            storage.save_drift_check(
                detector.name,
                score,
                detector.threshold,
                self.window_size,
                first_id,
                last_id,
                path=self.path,
            )
            if score <= detector.threshold:
                continue
            details = {
                "threshold": detector.threshold,
                "window_size": self.window_size,
                "first_request_id": first_id,
                "last_request_id": last_id,
                "top": top_drifting(detector, requests, sets),
            }
            alert_ids.append(storage.save_alert(detector.name, score, details, path=self.path))
        return alert_ids


def sae_sets(requests: Sequence[dict]) -> list[list[int]]:
    """Each request's stored top-k SAE features."""
    return [r["features"] for r in requests]


def example_prompts(cluster: int, requests: Sequence[dict], sets: Sequence[Sequence[int]]) -> str:
    """Describe a cluster, which has no label, by a few of the window's prompts in it."""
    examples = [
        r["prompt"][:EXAMPLE_CHARS] for r, s in zip(requests, sets, strict=True) if cluster in s
    ]
    return " | ".join(examples[:EXAMPLES_PER_CLUSTER])


def _baseline_and_threshold(path: Path, window_size: int) -> tuple[dict, float]:
    """Load a baseline and the threshold calibrated for it at this window size."""
    metadata, baseline = load_baseline(path)
    return baseline, load_threshold(threshold_path(path), window_size, metadata["created"])


def sae_detector(window_size: int) -> Detector:
    """The SAE detector: stored top-k features, labelled from Neuronpedia."""
    baseline, threshold = _baseline_and_threshold(baseline_path(), window_size)
    return Detector(
        "sae",
        baseline["counts"],
        baseline["prompt_count"],
        threshold,
        sae_sets,
        lambda feature_id, *_: get_label(feature_id),
    )


class WindowEmbedder:
    """Embeds each request once and keeps only the latest window's embeddings."""

    def __init__(self, embed: Callable[[list[str]], np.ndarray] = embed_prompts) -> None:
        self._embed = embed
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, requests: Sequence[dict]) -> np.ndarray:
        """One embedding row per request, in order; embeds only requests not seen before."""
        missing = [r for r in requests if r["id"] not in self._cache]
        if missing:
            rows = self._embed([r["prompt"] for r in missing])
            self._cache.update(zip((r["id"] for r in missing), rows, strict=True))
        self._cache = {r["id"]: self._cache[r["id"]] for r in requests}
        return np.stack([self._cache[r["id"]] for r in requests]).astype(float)


def embedding_detector(window_size: int, embedder: WindowEmbedder | None = None) -> Detector:
    """The embedding detector: scored by mean-embedding distance, explained by its 50 clusters."""
    cluster_path = embed_baseline_path()
    clusters, _ = _baseline_and_threshold(cluster_path, window_size)
    centroids = load_centroids(cluster_path)
    mean_path = mean_baseline_path()
    metadata, baseline_mean = load_mean_baseline(mean_path)
    threshold = load_threshold(threshold_path(mean_path), window_size, metadata["created"])
    embed = embedder or WindowEmbedder()

    def cluster_sets(requests: Sequence[dict]) -> list[list[int]]:
        return assign_clusters(embed(requests), centroids)

    def score(requests: Sequence[dict], _: Sequence[Sequence[int]]) -> float:
        return mean_distance(embed(requests), baseline_mean)

    return Detector(
        "embedding_mean",
        clusters["counts"],
        clusters["prompt_count"],
        threshold,
        cluster_sets,
        example_prompts,
        score,
    )


def load_drift_job() -> DriftJob:
    """Build the job with both detectors from their saved baselines and thresholds."""
    window_size = load_config().window_size
    return DriftJob([sae_detector(window_size), embedding_detector(window_size)], window_size)
