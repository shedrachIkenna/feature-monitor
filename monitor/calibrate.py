"""Calibrate the overall drift threshold on normal traffic that the baseline never saw.

normal_test is split in half with a fixed seed. Random windows from the calibration half are
scored against the baseline with Jensen-Shannon divergence, and the threshold is set at the
configured percentile of those scores. Windows from the check half then measure how often normal
traffic really exceeds it. Scoring the calibration windows themselves would give the percentile
back by construction.

Run with: uv run python -m monitor.calibrate
"""

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import chain
from pathlib import Path

import numpy as np

from monitor.baseline import POOL_DIR, K, baseline_path, load_baseline, read_pool
from monitor.config import DriftConfig, load_config
from monitor.drift import js_divergence
from monitor.extract import extract_features

POOL = "normal_test"
SWEEP_SIZES = (100, 250, 500, 1000)
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results" / "normal_jsd.json"


def threshold_path(baseline: Path) -> Path:
    """Return the threshold file that sits next to a baseline: <name>_threshold.json."""
    return baseline.with_name(f"{baseline.stem}_threshold.json")


def features_cache_path(baseline: Path) -> Path:
    """Return the (uncommitted) cache of per-prompt top-k features for the normal_test pool."""
    return baseline.with_name(f"{baseline.stem}_{POOL}_features.json")


def load_or_extract_features(prompts: Sequence[str], cache: Path, k: int = K) -> list[list[int]]:
    """Return each prompt's top-k feature IDs, reusing the cache if it matches prompts and k."""
    if cache.exists():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload["k"] == k and payload["prompt_count"] == len(prompts):
            return payload["features"]
    features = [[hit.feature_id for hit in extract_features(p, k)] for p in prompts]
    payload = {"k": k, "prompt_count": len(prompts), "features": features}
    cache.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return features


def split_halves(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle indices 0..n-1 with a seeded generator and return (calibration, check) halves."""
    order = np.random.default_rng(seed).permutation(n)
    return order[: n // 2], order[n // 2 :]


def window_scores(
    feature_sets: Sequence[Sequence[int]],
    indices: np.ndarray,
    baseline_counts: dict[int, int],
    window_size: int,
    n_windows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Score n_windows random windows (window_size prompts drawn without replacement)."""
    if window_size > len(indices):
        raise ValueError(f"window_size {window_size} exceeds the {len(indices)} prompts available")
    scores = np.empty(n_windows)
    for i in range(n_windows):
        chosen = rng.choice(indices, size=window_size, replace=False)
        counts = Counter(chain.from_iterable(feature_sets[j] for j in chosen))
        scores[i] = js_divergence(baseline_counts, counts)
    return scores


def calibrate(
    feature_sets: Sequence[Sequence[int]],
    baseline_counts: dict[int, int],
    window_size: int,
    percentile: float,
    n_windows: int,
    seed: int,
) -> dict:
    """Set the threshold on the calibration half and measure the exceed rate on the check half."""
    calibration, check = split_halves(len(feature_sets), seed)
    rng = np.random.default_rng([seed, window_size])
    calibration_scores = window_scores(
        feature_sets, calibration, baseline_counts, window_size, n_windows, rng
    )
    check_scores = window_scores(feature_sets, check, baseline_counts, window_size, n_windows, rng)
    threshold = float(np.percentile(calibration_scores, percentile))
    return {
        "window_size": window_size,
        "threshold": threshold,
        "check_exceed_rate": float((check_scores > threshold).mean()),
        "calibration_scores": calibration_scores.tolist(),
        "check_scores": check_scores.tolist(),
    }


def save_threshold(result: dict, config: DriftConfig, baseline_metadata: dict, path: Path) -> None:
    """Write the threshold with everything needed to tell whether it still applies."""
    payload = {
        "threshold": result["threshold"],
        "window_size": result["window_size"],
        "percentile": config.threshold_percentile,
        "calibration_windows": config.calibration_windows,
        "seed": config.seed,
        "pool": POOL,
        "check_exceed_rate": result["check_exceed_rate"],
        "baseline_created": baseline_metadata["created"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_threshold(path: Path, window_size: int, baseline_created: str) -> float:
    """Return the saved threshold; raise if it was set for another window size or baseline."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["window_size"] != window_size:
        raise ValueError(
            f"threshold was calibrated for window_size {payload['window_size']}, "
            f"not {window_size}; rerun monitor.calibrate"
        )
    if payload["baseline_created"] != baseline_created:
        raise ValueError("threshold was calibrated against a different baseline")
    return payload["threshold"]


def main() -> None:
    """Calibrate the configured window size, save it, and sweep other sizes for comparison."""
    config = load_config()
    base_path = baseline_path()
    metadata, baseline = load_baseline(base_path)
    prompts = read_pool(POOL_DIR / f"{POOL}.jsonl")
    feature_sets = load_or_extract_features(prompts, features_cache_path(base_path), metadata["k"])

    sizes = sorted(set(SWEEP_SIZES) | {config.window_size})
    results = {
        size: calibrate(
            feature_sets,
            baseline["counts"],
            size,
            config.threshold_percentile,
            config.calibration_windows,
            config.seed,
        )
        for size in sizes
    }

    chosen = results[config.window_size]
    path = threshold_path(base_path)
    save_threshold(chosen, config, metadata, path)
    if load_threshold(path, config.window_size, metadata["created"]) != chosen["threshold"]:
        raise RuntimeError("reloaded threshold differs from the one just saved")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps({"config": vars(config), "results": list(results.values())}) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {path} and {RESULTS_PATH}")
    print("window  median JSD  threshold  check-half exceed rate")
    for size, r in results.items():
        marker = "  <- configured" if size == config.window_size else ""
        print(
            f"{size:>6}  {np.median(r['calibration_scores']):10.4f}  {r['threshold']:9.4f}"
            f"  {r['check_exceed_rate']:6.1%}{marker}"
        )


if __name__ == "__main__":
    main()
