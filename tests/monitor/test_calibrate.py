from pathlib import Path

import numpy as np
import pytest

from monitor.baseline import sample_prompts
from monitor.calibrate import (
    calibrate,
    load_threshold,
    save_threshold,
    threshold_path,
    unused_prompts,
    window_scores,
)
from monitor.config import DriftConfig

N_FEATURES = 300
K = 5


def synthetic_feature_sets(n: int, seed: int) -> list[list[int]]:
    """Each prompt picks K distinct features with a fixed skewed popularity."""
    rng = np.random.default_rng(seed)
    weights = 1 / np.arange(1, N_FEATURES + 1)
    weights /= weights.sum()
    return [rng.choice(N_FEATURES, size=K, replace=False, p=weights).tolist() for _ in range(n)]


def counts_of(feature_sets: list[list[int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for features in feature_sets:
        for f in features:
            counts[f] = counts.get(f, 0) + 1
    return counts


def test_unused_prompts_are_exactly_those_outside_the_baseline_sample() -> None:
    pool = [f"p{i}" for i in range(50)]
    metadata = {"prompt_count": 20, "seed": 3}

    unused = unused_prompts(pool, metadata)

    assert len(unused) == 30
    assert set(unused).isdisjoint(sample_prompts(pool, 20, seed=3))
    assert unused == [p for p in pool if p in set(unused)]


def test_window_scores_reject_window_larger_than_half() -> None:
    with pytest.raises(ValueError):
        window_scores([[1]] * 4, np.arange(4), {1: 1}, 5, 1, np.random.default_rng(0))


def test_calibrate_is_reproducible_and_check_half_rarely_exceeds() -> None:
    calibration = synthetic_feature_sets(4000, seed=1)
    check = synthetic_feature_sets(4000, seed=3)
    baseline = counts_of(synthetic_feature_sets(3000, seed=2))

    result = calibrate(calibration, check, baseline, 100, 99, 500, seed=0)
    again = calibrate(calibration, check, baseline, 100, 99, 500, seed=0)

    assert result == again
    assert np.mean(np.array(result["calibration_scores"]) > result["threshold"]) <= 0.01
    assert result["check_exceed_rate"] < 0.05


def test_saved_threshold_reloads_and_rejects_mismatches(tmp_path: Path) -> None:
    config = DriftConfig(window_size=100, threshold_percentile=99, calibration_windows=10, seed=0)
    result = {"threshold": 0.12, "window_size": 100, "check_exceed_rate": 0.01}
    path = threshold_path(tmp_path / "gpt2_8.json")
    save_threshold(result, config, {"created": "t0"}, {"a": 1}, path)

    assert path.name == "gpt2_8_threshold.json"
    assert load_threshold(path, 100, "t0") == 0.12
    with pytest.raises(ValueError, match="window_size"):
        load_threshold(path, 200, "t0")
    with pytest.raises(ValueError, match="baseline"):
        load_threshold(path, 100, "t1")
