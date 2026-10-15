"""Measure how much the calibrated threshold and its held-out exceed rate move with the seed.

Each seed gives a different half-split of normal_test and different random windows. A swap run
at the configured seed sets the threshold on the check half and tests it on the calibration half.
Uses the per-prompt feature cache written by monitor.calibrate, so it needs no model.

Run with: uv run python -m eval.threshold_stability
"""

import json
from pathlib import Path

import numpy as np

from monitor.baseline import baseline_path, load_baseline
from monitor.calibrate import (
    SWEEP_SIZES,
    calibrate,
    features_cache_path,
    split_halves,
    window_scores,
)
from monitor.config import load_config

N_SEEDS = 20
OUT_PATH = Path(__file__).resolve().parent / "results" / "threshold_stability.json"


def swapped_exceed_rate(
    feature_sets: list[list[int]],
    baseline_counts: dict[int, int],
    window_size: int,
    percentile: float,
    n_windows: int,
    seed: int,
) -> float:
    """Set the threshold on the check half and return the exceed rate on the calibration half."""
    calibration, check = split_halves(len(feature_sets), seed)
    rng = np.random.default_rng([seed, window_size])
    first = window_scores(feature_sets, check, baseline_counts, window_size, n_windows, rng)
    second = window_scores(feature_sets, calibration, baseline_counts, window_size, n_windows, rng)
    return float((second > np.percentile(first, percentile)).mean())


def main() -> None:
    """Run every seed and window size, save the raw numbers and print a summary."""
    config = load_config()
    base_path = baseline_path()
    _, baseline = load_baseline(base_path)
    cache = features_cache_path(base_path)
    if not cache.exists():
        raise FileNotFoundError(f"{cache} is missing; run monitor.calibrate first")
    feature_sets = json.loads(cache.read_text(encoding="utf-8"))["features"]

    rows = []
    for size in SWEEP_SIZES:
        for seed in range(N_SEEDS):
            r = calibrate(
                feature_sets,
                baseline["counts"],
                size,
                config.threshold_percentile,
                config.calibration_windows,
                seed,
            )
            rows.append(
                {
                    "window_size": size,
                    "seed": seed,
                    "threshold": r["threshold"],
                    "check_exceed_rate": r["check_exceed_rate"],
                }
            )
    swaps = {
        size: swapped_exceed_rate(
            feature_sets,
            baseline["counts"],
            size,
            config.threshold_percentile,
            config.calibration_windows,
            config.seed,
        )
        for size in SWEEP_SIZES
    }

    OUT_PATH.write_text(
        json.dumps({"config": vars(config), "seeds": rows, "swap_exceed_rate": swaps}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUT_PATH}")
    print(f"{N_SEEDS} seeds per window size; exceed rate = held-out half above the threshold")
    print("window  threshold min-max    exceed mean  exceed min-max  seeds >2%  swap(seed 0)")
    for size in SWEEP_SIZES:
        t = np.array([r["threshold"] for r in rows if r["window_size"] == size])
        e = np.array([r["check_exceed_rate"] for r in rows if r["window_size"] == size])
        print(
            f"{size:>6}  {t.min():.4f}-{t.max():.4f}  {e.mean():10.1%}  "
            f"{e.min():5.1%}-{e.max():5.1%}  {int((e > 0.02).sum()):>9}  {swaps[size]:11.1%}"
        )


if __name__ == "__main__":
    main()
