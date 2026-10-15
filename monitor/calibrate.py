"""Calibrate the overall drift threshold on normal traffic that the baseline never saw.

The threshold is the configured percentile of Jensen-Shannon scores for random windows drawn from
the calibration pool: the ~11,000 normal_train prompts left out of the baseline sample. Windows
from a separate check pool (normal_test) then measure how often normal traffic really exceeds it;
scoring the calibration windows themselves would give the percentile back by construction.
Every window drawn from one finite pool shares that pool's small offset from the baseline, so a
large calibration pool and a separate check pool keep the measured false-alarm rate honest.

Run with: uv run python -m monitor.calibrate
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from monitor.baseline import (
    POOL_DIR,
    K,
    baseline_path,
    load_baseline,
    read_pool,
    sample_prompts,
)
from monitor.config import DriftConfig, load_config
from monitor.drift import score_window
from monitor.extract import extract_features

CALIBRATION_POOL = "normal_train_unused"
CHECK_POOL = "normal_test"
SWEEP_SIZES = (100, 250, 500, 1000)
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results" / "normal_jsd.json"


def threshold_path(baseline: Path) -> Path:
    """Return the threshold file that sits next to a baseline: <name>_threshold.json."""
    return baseline.with_name(f"{baseline.stem}_threshold.json")


def features_cache_path(baseline: Path, pool: str) -> Path:
    """Return the (uncommitted) cache of per-prompt top-k features for one pool."""
    return baseline.with_name(f"{baseline.stem}_{pool}_features.json")


def unused_prompts(pool: list[str], baseline_metadata: dict) -> list[str]:
    """Return the pool prompts not drawn into the baseline sample, in pool order."""
    used = set(sample_prompts(pool, baseline_metadata["prompt_count"], baseline_metadata["seed"]))
    return [p for p in pool if p not in used]


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
        scores[i] = score_window(baseline_counts, (feature_sets[j] for j in chosen))
    return scores


def calibrate(
    calibration_sets: Sequence[Sequence[int]],
    check_sets: Sequence[Sequence[int]],
    baseline_counts: dict[int, int],
    window_size: int,
    percentile: float,
    n_windows: int,
    seed: int,
) -> dict:
    """Set the threshold on the calibration pool and measure the exceed rate on the check pool."""
    rng = np.random.default_rng([seed, window_size])
    calibration_scores = window_scores(
        calibration_sets,
        np.arange(len(calibration_sets)),
        baseline_counts,
        window_size,
        n_windows,
        rng,
    )
    check_scores = window_scores(
        check_sets, np.arange(len(check_sets)), baseline_counts, window_size, n_windows, rng
    )
    threshold = float(np.percentile(calibration_scores, percentile))
    return {
        "window_size": window_size,
        "threshold": threshold,
        "check_exceed_rate": float((check_scores > threshold).mean()),
        "calibration_scores": calibration_scores.tolist(),
        "check_scores": check_scores.tolist(),
    }


def calibrate_sizes(
    calibration_sets: Sequence[Sequence[int]],
    check_sets: Sequence[Sequence[int]],
    baseline_counts: dict[int, int],
    config: DriftConfig,
) -> dict[int, dict]:
    """Calibrate the configured window size and every sweep size, keyed by window size."""
    sizes = sorted(set(SWEEP_SIZES) | {config.window_size})
    return {
        size: calibrate(
            calibration_sets,
            check_sets,
            baseline_counts,
            size,
            config.threshold_percentile,
            config.calibration_windows,
            config.seed,
        )
        for size in sizes
    }


def save_threshold(
    result: dict,
    config: DriftConfig,
    baseline_metadata: dict,
    pool_sizes: dict[str, int],
    path: Path,
) -> None:
    """Write the threshold with everything needed to tell whether it still applies."""
    payload = {
        "threshold": result["threshold"],
        "window_size": result["window_size"],
        "percentile": config.threshold_percentile,
        "calibration_windows": config.calibration_windows,
        "seed": config.seed,
        "pool_sizes": pool_sizes,
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


def print_sweep(results: dict[int, dict], configured_size: int) -> None:
    """Print median score, threshold and check-pool exceed rate for each window size."""
    print("window  median JSD  threshold  check-pool exceed rate")
    for size, r in results.items():
        marker = "  <- configured" if size == configured_size else ""
        print(
            f"{size:>6}  {np.median(r['calibration_scores']):10.4f}  {r['threshold']:9.4f}"
            f"  {r['check_exceed_rate']:6.1%}{marker}"
        )


def pool_prompts(baseline_metadata: dict) -> dict[str, list[str]]:
    """Return the calibration pool (train prompts outside the baseline) and the check pool."""
    train = read_pool(POOL_DIR / f"{baseline_metadata['pool']}.jsonl")
    return {
        CALIBRATION_POOL: unused_prompts(train, baseline_metadata),
        CHECK_POOL: read_pool(POOL_DIR / f"{CHECK_POOL}.jsonl"),
    }


def load_pool_features(base_path: Path, baseline_metadata: dict) -> dict[str, list[list[int]]]:
    """Return per-prompt features for the calibration and check pools, extracting if needed."""
    return {
        name: load_or_extract_features(
            prompts, features_cache_path(base_path, name), baseline_metadata["k"]
        )
        for name, prompts in pool_prompts(baseline_metadata).items()
    }


def main() -> None:
    """Calibrate the configured window size, save it, and sweep other sizes for comparison."""
    config = load_config()
    base_path = baseline_path()
    metadata, baseline = load_baseline(base_path)
    features = load_pool_features(base_path, metadata)

    results = calibrate_sizes(
        features[CALIBRATION_POOL], features[CHECK_POOL], baseline["counts"], config
    )

    chosen = results[config.window_size]
    path = threshold_path(base_path)
    pool_sizes = {name: len(sets) for name, sets in features.items()}
    save_threshold(chosen, config, metadata, pool_sizes, path)
    if load_threshold(path, config.window_size, metadata["created"]) != chosen["threshold"]:
        raise RuntimeError("reloaded threshold differs from the one just saved")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"config": vars(config), "pool_sizes": pool_sizes, "results": list(results.values())}
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {path} and {RESULTS_PATH}")
    print(f"Pools: {pool_sizes}")
    print_sweep(results, config.window_size)


if __name__ == "__main__":
    main()
