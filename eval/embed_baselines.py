"""Stronger embedding baselines, evaluated offline exactly like eval/multiseed.py.

The live embedding detector keeps one of 50 k-means clusters per prompt. Three variants keep more
of the same all-MiniLM-L6-v2 embeddings:
- k200: the same method with 200 clusters, fitted the same way on the same 3,000 baseline prompts;
- mean: squared distance between the window's mean embedding and the baseline's mean embedding;
- mmd: the (biased) squared maximum mean discrepancy between window and baseline embeddings with
  a Gaussian kernel, width set by the median heuristic on the baseline (median pairwise distance).

Each variant's threshold is the configured percentile of its scores on the same 1,000 random
normal_train windows that calibrated the live detectors (same pool, seed and draw order), and each
is replayed through the same scenarios and seeds, with the same threshold sweep.

Run with: uv run python -m eval.embed_baselines   (after eval.multiseed)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.multiseed import (
    CACHE_PATH,
    FILE_SCENARIOS,
    NORMAL,
    SEEDS,
    SWEEP_PERCENTILES,
    false_alarms,
    load_pools,
    load_scenarios,
    rolling_scores,
    sequences,
    sweep,
)
from monitor.baseline import baseline_path, load_baseline
from monitor.calibrate import CALIBRATION_POOL, CHECK_POOL
from monitor.config import load_config
from traffic.generate import load_pool

ROOT = Path(__file__).resolve().parent.parent
EMBED_CACHE = CACHE_PATH.with_name("embeddings.npz")
OUT_PATH = ROOT / "eval" / "results" / "embed_baselines.json"
VARIANTS = ("k200", "mean", "mmd")
N_CLUSTERS = 200
KMEANS_SEED = 0
# Pairs sampled for the median heuristic; all 4.5 million baseline pairs give the same answer
# to three digits but take longer.
MEDIAN_PAIRS = 200_000

Scorer = Callable[[np.ndarray], np.ndarray]


def calibration_windows(
    pool_size: int, window_size: int, n_windows: int, seed: int
) -> list[np.ndarray]:
    """The index sets monitor.calibrate draws first: same generator, same order."""
    rng = np.random.default_rng([seed, window_size])
    indices = np.arange(pool_size)
    return [rng.choice(indices, size=window_size, replace=False) for _ in range(n_windows)]


def mean_distance(window: np.ndarray, baseline_mean: np.ndarray) -> float:
    """Squared distance between the window's mean embedding and the baseline's."""
    return float(((window.mean(axis=0) - baseline_mean) ** 2).sum())


def rolling_mean_distance(
    embeddings: np.ndarray, baseline_mean: np.ndarray, window_size: int
) -> np.ndarray:
    """mean_distance of every full window, element i covering rows i .. i+window_size-1."""
    sums = np.vstack([np.zeros(embeddings.shape[1]), np.cumsum(embeddings, axis=0, dtype=float)])
    means = (sums[window_size:] - sums[:-window_size]) / window_size
    return ((means - baseline_mean) ** 2).sum(axis=1)


def gaussian_kernel(a: np.ndarray, b: np.ndarray, width: float) -> np.ndarray:
    """exp(-|x - y|^2 / (2 width^2)) for every row x of a and y of b."""
    squared = (a**2).sum(axis=1)[:, None] + (b**2).sum(axis=1)[None, :] - 2 * a @ b.T
    return np.exp(-np.maximum(squared, 0.0) / (2 * width**2))


def median_width(baseline: np.ndarray, seed: int, pairs: int = MEDIAN_PAIRS) -> float:
    """Median distance between distinct baseline embeddings, from a seeded sample of pairs."""
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(baseline), pairs)
    j = rng.integers(0, len(baseline), pairs)
    keep = i != j
    return float(np.median(np.linalg.norm(baseline[i[keep]] - baseline[j[keep]], axis=1)))


class MMD:
    """Biased squared MMD against a fixed baseline: mean k(W,W) + mean k(B,B) - 2 mean k(W,B)."""

    def __init__(self, baseline: np.ndarray, width: float) -> None:
        self.baseline = baseline
        self.width = width
        self.baseline_term = float(gaussian_kernel(baseline, baseline, width).mean())

    def cross(self, embeddings: np.ndarray) -> np.ndarray:
        """Each row's mean kernel value against the baseline."""
        return gaussian_kernel(embeddings, self.baseline, self.width).mean(axis=1)

    def score(self, window: np.ndarray) -> float:
        """MMD^2 of one window, computed directly."""
        within = gaussian_kernel(window, window, self.width).mean()
        return float(within + self.baseline_term - 2 * self.cross(window).mean())

    def rolling(self, embeddings: np.ndarray, window_size: int) -> np.ndarray:
        """score() of every full window, from 2-D prefix sums of the run's kernel matrix."""
        gram = gaussian_kernel(embeddings, embeddings, self.width)
        table = np.zeros((len(gram) + 1, len(gram) + 1))
        table[1:, 1:] = gram.cumsum(axis=0).cumsum(axis=1)
        s = np.arange(len(gram) - window_size + 1)
        e = s + window_size
        within = table[e, e] - table[s, e] - table[e, s] + table[s, s]
        cross = np.concatenate([[0.0], np.cumsum(self.cross(embeddings))])
        return (
            within / window_size**2 + self.baseline_term - 2 * (cross[e] - cross[s]) / window_size
        )


def embeddings_for(prompts: Sequence[str], cache: Path) -> dict[str, np.ndarray]:
    """MiniLM embedding of each prompt, embedding only those not in the cache."""
    from monitor.embed_drift import embed_prompts

    stored: dict[str, np.ndarray] = {}
    if cache.exists():
        with np.load(cache) as data:
            stored = dict(zip(data["prompts"].tolist(), data["embeddings"], strict=True))
    missing = sorted(set(prompts) - set(stored))
    if missing:
        print(f"Embedding {len(missing)} new prompts ...", flush=True)
        stored.update(zip(missing, embed_prompts(missing), strict=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, prompts=np.array(list(stored)), embeddings=np.stack(list(stored.values())))
    return {p: stored[p] for p in prompts}


def main() -> None:
    """Calibrate and replay the three variants; save sweeps and every window's score."""
    from sklearn.cluster import KMeans

    from monitor.embed_drift import assign_clusters, cluster_counts, embeddings_cache_path

    config = load_config()
    window_size = config.window_size
    baseline = np.load(embeddings_cache_path("baseline")).astype(float)
    calib = np.load(embeddings_cache_path(CALIBRATION_POOL)).astype(float)
    metadata, _ = load_baseline(baseline_path())
    if len(baseline) != metadata["prompt_count"]:
        raise ValueError("baseline embedding cache does not match the SAE baseline")
    windows = calibration_windows(len(calib), window_size, config.calibration_windows, config.seed)

    scenarios = load_scenarios()
    names = {s.normal_pool for s in scenarios.values()}
    names |= {s.shift_pool for s in scenarios.values() if s.shift_pool}
    runs = sequences(scenarios, SEEDS, load_pools(names))
    # normal_test embeddings are already cached by monitor.embed_drift; reuse them.
    known = dict(
        zip(load_pool(CHECK_POOL), np.load(embeddings_cache_path(CHECK_POOL)), strict=True)
    )
    needed = sorted({r["prompt"] for rows in runs.values() for r in rows} - set(known))
    known.update(embeddings_for(needed, EMBED_CACHE))
    run_embeddings = {
        key: np.array([known[r["prompt"]] for r in rows], dtype=float) for key, rows in runs.items()
    }

    centroids = (
        KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=KMEANS_SEED)
        .fit(baseline)
        .cluster_centers_
    )
    counts = cluster_counts(assign_clusters(baseline, centroids))
    calib_sets = assign_clusters(calib, centroids)
    baseline_mean = baseline.mean(axis=0)
    width = median_width(baseline, config.seed)
    mmd = MMD(baseline, width)

    calibrate: dict[str, Callable[[np.ndarray], float]] = {
        "k200": lambda idx: rolling_scores([calib_sets[i] for i in idx], counts, window_size)[0],
        "mean": lambda idx: mean_distance(calib[idx], baseline_mean),
        "mmd": lambda idx: mmd.score(calib[idx]),
    }
    replay: dict[str, Scorer] = {
        "k200": lambda e: rolling_scores(assign_clusters(e, centroids), counts, window_size),
        "mean": lambda e: rolling_mean_distance(e, baseline_mean, window_size),
        "mmd": lambda e: mmd.rolling(e, window_size),
    }

    results, all_scores = {}, {}
    for variant in VARIANTS:
        print(f"Scoring {variant} ...", flush=True)
        calib_scores = np.array([calibrate[variant](idx) for idx in windows])
        thresholds = {p: float(np.percentile(calib_scores, p)) for p in SWEEP_PERCENTILES}
        saved = thresholds[config.threshold_percentile]
        scores = {key: replay[variant](e) for key, e in run_embeddings.items()}
        results[variant] = {
            "saved_threshold": saved,
            "calibration_median": float(np.median(calib_scores)),
            "false_alarms": false_alarms([s for (n, _), s in scores.items() if n == NORMAL], saved),
            "sweep": sweep(scores, runs, thresholds, window_size),
        }
        all_scores[variant] = {
            f"{n}/{s}": [float(f"{x:.8g}") for x in v] for (n, s), v in scores.items()
        }

    OUT_PATH.write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "config": vars(config),
                "seeds": list(SEEDS),
                "scenarios": sorted(scenarios),
                "file_scenarios": list(FILE_SCENARIOS),
                "n_clusters": N_CLUSTERS,
                "kernel_width": width,
                "detectors": results,
                "scores": all_scores,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUT_PATH}")
    for variant, r in results.items():
        fa = r["false_alarms"]
        print(
            f"\n{variant}: threshold {r['saved_threshold']:.6g} (normal median"
            f" {r['calibration_median']:.6g}); normal_only windows above {fa['window_rate']:.2%},"
            f" runs with an alert {fa['run_rate']:.0%}"
        )
        print("  percentile  FA windows  FA runs  median delay per scenario")
        for point in r["sweep"]:
            delays = "  ".join(
                f"{k.removesuffix('_ramp')} {point[k]['median']}"
                for k in sorted(k for k in point if k.endswith("_ramp"))
            )
            print(
                f"  {point['percentile']:>10}  {point['window_rate']:9.2%}  {point['run_rate']:6.0%}"
                f"   {delays}"
            )


if __name__ == "__main__":
    main()
