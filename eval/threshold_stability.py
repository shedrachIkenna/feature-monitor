"""Measure how much the calibrated threshold and its held-out exceed rate move with the seed.

For each seed, the threshold is calibrated on the unused normal_train pool and checked on all of
normal_test. It is also checked on a random 3,000-prompt subset of normal_test, which mimics a
fresh batch of normal traffic with its own small offset from the baseline. Uses the feature caches
written by monitor.calibrate, so it needs no model.

Run with: uv run python -m eval.threshold_stability
"""

import json
from pathlib import Path

import numpy as np

from monitor.baseline import baseline_path, load_baseline
from monitor.calibrate import (
    CALIBRATION_POOL,
    CHECK_POOL,
    SWEEP_SIZES,
    calibrate,
    load_pool_features,
    window_scores,
)
from monitor.config import load_config

N_SEEDS = 20
SUBSET_SIZE = 3000
OUT_PATH = Path(__file__).resolve().parent / "results" / "threshold_stability.json"


def main() -> None:
    """Run every seed and window size, save the raw numbers and print a summary."""
    config = load_config()
    base_path = baseline_path()
    metadata, baseline = load_baseline(base_path)
    features = load_pool_features(base_path, metadata)
    check = features[CHECK_POOL]

    rows = []
    for size in SWEEP_SIZES:
        for seed in range(N_SEEDS):
            r = calibrate(
                features[CALIBRATION_POOL],
                check,
                baseline["counts"],
                size,
                config.threshold_percentile,
                config.calibration_windows,
                seed,
            )
            rng = np.random.default_rng([seed, size, SUBSET_SIZE])
            subset = rng.choice(len(check), size=SUBSET_SIZE, replace=False)
            subset_scores = window_scores(
                check, subset, baseline["counts"], size, config.calibration_windows, rng
            )
            rows.append(
                {
                    "window_size": size,
                    "seed": seed,
                    "threshold": r["threshold"],
                    "check_exceed_rate": r["check_exceed_rate"],
                    "subset_exceed_rate": float((subset_scores > r["threshold"]).mean()),
                }
            )

    OUT_PATH.write_text(
        json.dumps({"config": vars(config), "subset_size": SUBSET_SIZE, "seeds": rows}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {OUT_PATH}")
    print(f"{N_SEEDS} seeds per window size; exceed rate = normal windows above the threshold")
    print(
        "window  threshold min-max  full normal_test: mean, min-max, >2%"
        f"  |  {SUBSET_SIZE}-prompt subsets: mean, min-max, >2%"
    )
    for size in SWEEP_SIZES:
        sel = [r for r in rows if r["window_size"] == size]
        t = np.array([r["threshold"] for r in sel])
        line = f"{size:>6}  {t.min():.4f}-{t.max():.4f}"
        for key in ("check_exceed_rate", "subset_exceed_rate"):
            e = np.array([r[key] for r in sel])
            line += (
                f"  |  {e.mean():5.1%}, {e.min():4.1%}-{e.max():5.1%}, {int((e > 0.02).sum()):>2}"
            )
        print(line)


if __name__ == "__main__":
    main()
