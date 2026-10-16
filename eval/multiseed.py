"""Offline multi-seed evaluation: replay every scenario with many seeds and score every window.

A prompt's top-k SAE features and its embedding cluster do not depend on when it is sent, so each
unique prompt is encoded once (cached) and any scenario can be replayed with any seed without the
server. Each run is scored after every request from the first full window onwards, with the same
Jensen-Shannon score as the live drift job, instead of once a minute.

Before anything else, the seed-42 replay is checked against eval/results/first_run.json: the
offline score at every request where the live job checked must match the stored score.

Measurements:
- detection delay: requests after the shift starts (request SHIFT_START) until the first window
  above the threshold, per seed, and the share of shifted prompts in that window;
- false alarms on normal_only: the share of windows above the threshold and the share of runs with
  at least one alert;
- a threshold sweep over percentiles of each detector's saved calibration scores, so the two
  detectors can be compared at equal false-alarm rates.

medical_nowhat_ramp is medical_ramp with the medical prompts that start with "What" removed: 66% of
medical prompts start with "What" against about 2% of normal ones, so this separates the topic from
that one wording cue.

Run with: uv run python -m eval.multiseed
"""

from __future__ import annotations

import dataclasses
import json
import statistics
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from monitor.baseline import K, baseline_path, load_baseline
from monitor.calibrate import CHECK_POOL, features_cache_path, threshold_path
from monitor.config import load_config
from traffic.generate import Scenario, build_sequence, load_pool, load_scenario

ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "traffic" / "scenarios"
CACHE_PATH = ROOT / "data" / "multiseed" / "representations.json"
OUT_PATH = ROOT / "eval" / "results" / "multiseed.json"
FIRST_RUN_PATH = ROOT / "eval" / "results" / "first_run.json"
RESULTS_DIR = ROOT / "eval" / "results"

SEEDS = tuple(range(10))
FIRST_RUN_SEED = 42
NORMAL = "normal_only"
FILE_SCENARIOS = ("normal_only", "medical_ramp", "new_topic_ramp", "injection_ramp")
NOWHAT = "medical_nowhat_ramp"
NOWHAT_POOL = "shift_medical_nowhat"
SHIFT_START = 500  # every shift scenario holds 0% shift for its first 500 requests
SWEEP_PERCENTILES = (90.0, 95.0, 97.5, 99.0, 99.5, 99.9, 100.0)
DETECTORS = ("sae", "embedding")
# The live job and the replay sum the same terms in a different order.
FAITHFUL_TOLERANCE = 1e-9


def starts_with_what(text: str) -> bool:
    """True for prompts whose first word starts with "what" (What, What's, what ...)."""
    return text.lstrip().casefold().startswith("what")


def load_pools(names: set[str]) -> dict[str, list[str]]:
    """Read the named pools; NOWHAT_POOL is shift_medical without its "What" prompts."""
    pools = {}
    for name in names:
        if name == NOWHAT_POOL:
            pools[name] = [t for t in load_pool("shift_medical") if not starts_with_what(t)]
        else:
            pools[name] = load_pool(name)
    return pools


def load_scenarios() -> dict[str, Scenario]:
    """The four scenario files plus medical_nowhat_ramp, which differs only in its shift pool."""
    scenarios = {name: load_scenario(SCENARIO_DIR / f"{name}.yaml") for name in FILE_SCENARIOS}
    scenarios[NOWHAT] = dataclasses.replace(
        scenarios["medical_ramp"], name=NOWHAT, shift_pool=NOWHAT_POOL
    )
    return scenarios


def sequences(
    scenarios: dict[str, Scenario], seeds: Sequence[int], pools: dict[str, list[str]]
) -> dict[tuple[str, int], list[dict]]:
    """The exact prompt sequence of every (scenario, seed) run."""
    return {
        (name, seed): build_sequence(dataclasses.replace(scenario, seed=seed), pools)
        for name, scenario in scenarios.items()
        for seed in seeds
    }


def rolling_scores(
    sets: Sequence[Sequence[int]], baseline_counts: dict[int, int], window_size: int
) -> np.ndarray:
    """JSD of every full window: element i scores requests i+1 .. i+window_size (1-based).

    Equal to monitor.drift.score_window on each window; IDs absent from both sides add nothing.
    """
    if len(sets) < window_size:
        return np.empty(0)
    size = 1 + max(max(baseline_counts), max((i for s in sets for i in s), default=0))
    base = np.zeros(size)
    for feature_id, count in baseline_counts.items():
        base[feature_id] = count
    arrays = [np.asarray(s, dtype=int) for s in sets]
    counts = np.zeros(size)
    for ids in arrays[:window_size]:
        np.add.at(counts, ids, 1)
    scores = [jensenshannon(base, counts, base=2) ** 2]
    for i in range(window_size, len(arrays)):
        np.add.at(counts, arrays[i], 1)
        np.subtract.at(counts, arrays[i - window_size], 1)
        scores.append(jensenshannon(base, counts, base=2) ** 2)
    return np.array(scores)


def first_alert(scores: np.ndarray, threshold: float, window_size: int) -> int | None:
    """Request number (1-based) of the first window scoring above threshold, or None."""
    above = np.flatnonzero(scores > threshold)
    return int(above[0]) + window_size if above.size else None


def shift_share(rows: Sequence[dict], request_number: int, window_size: int) -> float:
    """Share of shifted prompts in the window ending at request_number."""
    window = rows[request_number - window_size : request_number]
    return sum(r["is_shift"] for r in window) / len(window)


def detection(rows: Sequence[dict], scores: np.ndarray, threshold: float, window_size: int) -> dict:
    """First alert of one shift run: request number, delay after SHIFT_START, shift share."""
    number = first_alert(scores, threshold, window_size)
    if number is None:
        return {"first_alert": None, "delay": None, "shift_share": None}
    return {
        "first_alert": number,
        "delay": number - SHIFT_START,
        "shift_share": shift_share(rows, number, window_size),
    }


def mark_normal_first(found: dict, normal_first: int | None) -> dict:
    """Flag a detection that the same seed's normal_only run matched or beat.

    Every scenario with the same seed draws its normal prompts in the same order, so if normal
    traffic alone alerted by then, the alert cannot be credited to the shift. The match is
    approximate after SHIFT_START, where shifted prompts push the normal ones later.
    """
    first = found["first_alert"]
    return {
        **found,
        "normal_alert_first": first is not None
        and normal_first is not None
        and normal_first <= first,
    }


def false_alarms(runs: Sequence[np.ndarray], threshold: float) -> dict:
    """Share of normal windows above threshold, and share of normal runs with any alert."""
    return {
        "window_rate": float(np.mean(np.concatenate(runs) > threshold)),
        "run_rate": float(np.mean([bool((r > threshold).any()) for r in runs])),
    }


def delay_summary(delays: Sequence[int | None]) -> dict:
    """Median, min and max delay over the runs that alerted, and how many did."""
    found = [d for d in delays if d is not None]
    return {
        "detected": len(found),
        "runs": len(delays),
        # The median over every run is a real delay only when more than half of them alerted.
        "median": statistics.median(d if d is not None else float("inf") for d in delays)
        if len(found) * 2 > len(delays)
        else None,
        "min": min(found) if found else None,
        "max": max(found) if found else None,
    }


def sweep(
    scores: dict[tuple[str, int], np.ndarray],
    rows: dict[tuple[str, int], list[dict]],
    thresholds: dict[float, float],
    window_size: int,
) -> list[dict]:
    """For each threshold: false alarms on normal_only and the delay summary per shift scenario."""
    names = sorted({name for name, _ in scores if name != NORMAL})
    normal_runs = [s for (name, _), s in scores.items() if name == NORMAL]
    points = []
    for percentile, threshold in thresholds.items():
        point = {
            "percentile": percentile,
            "threshold": threshold,
            **false_alarms(normal_runs, threshold),
        }
        for name in names:
            delays = [
                detection(rows[key], s, threshold, window_size)["delay"]
                for key, s in scores.items()
                if key[0] == name
            ]
            point[name] = delay_summary(delays)
        points.append(point)
    return points


Encoder = Callable[[list[str]], tuple[list[list[int]], list[int]]]


def representations(prompts: set[str], cache: Path, encode: Encoder) -> dict[str, dict]:
    """Each prompt's top-k SAE features and embedding cluster, encoding only uncached prompts."""
    stored = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    missing = sorted(p for p in prompts if p not in stored)
    if missing:
        features, clusters = encode(missing)
        for prompt, f, c in zip(missing, features, clusters, strict=True):
            stored[prompt] = {"sae": f, "cluster": c}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(stored, ensure_ascii=False) + "\n", encoding="utf-8")
    return {p: stored[p] for p in prompts}


def seed_cache_from_calibration(cache: Path, centroids: np.ndarray) -> None:
    """Start the cache from normal_test's calibration caches, so those prompts are not re-encoded."""
    if cache.exists():
        return
    from monitor.embed_drift import assign_clusters, embeddings_cache_path

    prompts = load_pool(CHECK_POOL)
    features = json.loads(
        features_cache_path(baseline_path(), CHECK_POOL).read_text(encoding="utf-8")
    )
    embeddings = np.load(embeddings_cache_path(CHECK_POOL))
    if features["k"] != K or not len(features["features"]) == len(prompts) == len(embeddings):
        raise ValueError("normal_test calibration caches do not match the pool; rerun calibration")
    clusters = assign_clusters(embeddings, centroids)
    stored = {
        p: {"sae": f, "cluster": c[0]}
        for p, f, c in zip(prompts, features["features"], clusters, strict=True)
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(stored, ensure_ascii=False) + "\n", encoding="utf-8")


def model_encoder(centroids: np.ndarray) -> Encoder:
    """Encode prompts with GPT-2's SAE (top K features) and MiniLM (nearest centroid)."""
    from monitor.embed_drift import assign_clusters, embed_prompts
    from monitor.extract import extract_features

    def encode(prompts: list[str]) -> tuple[list[list[int]], list[int]]:
        print(f"Encoding {len(prompts)} new prompts ...", flush=True)
        features = [[h.feature_id for h in extract_features(p, K)] for p in prompts]
        clusters = [c[0] for c in assign_clusters(embed_prompts(prompts), centroids)]
        return features, clusters

    return encode


def score_runs(
    runs: dict[tuple[str, int], list[dict]],
    reps: dict[str, dict],
    baselines: dict[str, dict[int, int]],
    window_size: int,
) -> dict[str, dict[tuple[str, int], np.ndarray]]:
    """Rolling scores of every run, per detector."""
    sets = {
        "sae": lambda r: reps[r["prompt"]]["sae"],
        "embedding": lambda r: [reps[r["prompt"]]["cluster"]],
    }
    return {
        detector: {
            key: rolling_scores([sets[detector](r) for r in rows], baselines[detector], window_size)
            for key, rows in runs.items()
        }
        for detector in DETECTORS
    }


def faithfulness(
    scores: dict[str, dict[tuple[str, int], np.ndarray]], first_run: dict, window_size: int
) -> dict:
    """Largest gap between replayed and live scores at the first run's check points."""
    gaps: dict[str, list[float]] = {d: [] for d in DETECTORS}
    for scenario in first_run["scenarios"]:
        for check in scenario["checks"]:
            for detector, live in check["scores"].items():
                replay = scores[detector][(scenario["scenario"], FIRST_RUN_SEED)]
                gaps[detector].append(abs(replay[check["request_number"] - window_size] - live))
    return {d: {"checks": len(g), "max_abs_diff": max(g)} for d, g in gaps.items()}


def main() -> None:
    """Check the replay against the first run, then replay every scenario with every seed."""
    from monitor.embed_drift import embed_baseline_path, load_centroids

    config = load_config()
    window_size = config.window_size
    sae_path, embed_path = baseline_path(), embed_baseline_path()
    paths = {"sae": sae_path, "embedding": embed_path}
    baselines = {d: load_baseline(p)[1]["counts"] for d, p in paths.items()}
    saved = {
        d: json.loads(threshold_path(p).read_text(encoding="utf-8"))["threshold"]
        for d, p in paths.items()
    }
    centroids = load_centroids(embed_path)

    scenarios = load_scenarios()
    names = {s.normal_pool for s in scenarios.values()}
    names |= {s.shift_pool for s in scenarios.values() if s.shift_pool}
    pools = load_pools(names)
    live = {k: v for k, v in scenarios.items() if k in FILE_SCENARIOS}
    replay = sequences(live, [FIRST_RUN_SEED], pools)
    runs = sequences(scenarios, SEEDS, pools)

    seed_cache_from_calibration(CACHE_PATH, centroids)
    prompts = {r["prompt"] for rows in (*replay.values(), *runs.values()) for r in rows}
    reps = representations(prompts, CACHE_PATH, model_encoder(centroids))

    first_run = json.loads(FIRST_RUN_PATH.read_text(encoding="utf-8"))
    check = faithfulness(score_runs(replay, reps, baselines, window_size), first_run, window_size)
    print(f"Replay vs first run (seed {FIRST_RUN_SEED}): {check}")
    if any(c["max_abs_diff"] > FAITHFUL_TOLERANCE for c in check.values()):
        raise RuntimeError("replayed scores differ from the live first run; not trusting replay")

    scores = score_runs(runs, reps, baselines, window_size)
    calibration = {
        "sae": RESULTS_DIR / "normal_jsd.json",
        "embedding": RESULTS_DIR / "embed_jsd.json",
    }
    results = {}
    for detector in DETECTORS:
        payload = json.loads(calibration[detector].read_text(encoding="utf-8"))
        calib = next(r for r in payload["results"] if r["window_size"] == window_size)
        thresholds = {
            p: float(np.percentile(calib["calibration_scores"], p)) for p in SWEEP_PERCENTILES
        }
        if not np.isclose(thresholds[config.threshold_percentile], saved[detector]):
            raise RuntimeError(f"{detector}: calibration scores do not reproduce saved threshold")
        thresholds[config.threshold_percentile] = saved[detector]
        mine = scores[detector]
        results[detector] = {
            "saved_threshold": saved[detector],
            "false_alarms": false_alarms(
                [s for (n, _), s in mine.items() if n == NORMAL], saved[detector]
            ),
            "detections": {
                f"{name}/{seed}": mark_normal_first(
                    detection(runs[(name, seed)], s, saved[detector], window_size),
                    first_alert(mine[(NORMAL, seed)], saved[detector], window_size),
                )
                for (name, seed), s in mine.items()
                if name != NORMAL
            },
            "sweep": sweep(mine, runs, thresholds, window_size),
        }

    OUT_PATH.write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "config": vars(config),
                "seeds": list(SEEDS),
                "shift_start": SHIFT_START,
                "scenarios": sorted(scenarios),
                "nowhat_pool_size": len(pools[NOWHAT_POOL]),
                "faithfulness": check,
                "detectors": results,
                # Every window's score, so the analysis can be redone without the model.
                "scores": {
                    d: {f"{n}/{s}": np.round(v, 6).tolist() for (n, s), v in scores[d].items()}
                    for d in DETECTORS
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUT_PATH}")
    print_summary(results)


def print_summary(results: dict) -> None:
    """Print false alarms and delay summaries at the saved thresholds."""
    for detector, r in results.items():
        fa = r["false_alarms"]
        print(
            f"\n{detector}: threshold {r['saved_threshold']:.4f}; normal_only windows above"
            f" {fa['window_rate']:.2%}, runs with an alert {fa['run_rate']:.0%}"
        )
        saved = next(p for p in r["sweep"] if p["threshold"] == r["saved_threshold"])
        for name in sorted(k for k in saved if k.endswith("_ramp")):
            s = saved[name]
            flagged = sorted(
                k.split("/")[1]
                for k, v in r["detections"].items()
                if k.startswith(f"{name}/") and v["normal_alert_first"]
            )
            print(
                f"  {name:<20} detected {s['detected']}/{s['runs']}  median delay {s['median']}"
                f"  range {s['min']}-{s['max']}  normal alerted first in seeds {flagged or '-'}"
            )
        print("  percentile  threshold  FA windows  FA runs  median delay per scenario")
        for point in r["sweep"]:
            delays = "  ".join(
                f"{k.removesuffix('_ramp')} {point[k]['median']}"
                for k in sorted(k for k in point if k.endswith("_ramp"))
            )
            print(
                f"  {point['percentile']:>10}  {point['threshold']:.4f}   {point['window_rate']:8.2%}"
                f"  {point['run_rate']:6.0%}   {delays}"
            )


if __name__ == "__main__":
    main()
