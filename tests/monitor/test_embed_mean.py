import json

import numpy as np
import pytest

from eval.embed_baselines import calibration_windows
from monitor.embed_mean import calibrate_mean, load_mean_baseline, mean_distance


def rows(n: int, seed: int, shift: float = 0.0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, 4)) + shift


def test_mean_distance_is_squared_distance_between_means() -> None:
    window = np.array([[0.0, 0.0], [2.0, 0.0]])
    assert mean_distance(window, np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_calibrate_mean_uses_the_calibration_windows_of_monitor_calibrate() -> None:
    calibration, check, base = rows(80, 0), rows(80, 1), rows(50, 2).mean(axis=0)

    result = calibrate_mean(calibration, check, base, 10, 90.0, 20, seed=3)

    windows = calibration_windows(80, 10, 20, seed=3)
    expected = np.percentile([mean_distance(calibration[w], base) for w in windows], 90.0)
    assert result["threshold"] == expected
    assert result["window_size"] == 10
    assert 0.0 <= result["check_exceed_rate"] <= 1.0


def test_check_windows_from_a_shifted_pool_exceed_the_threshold() -> None:
    calibration, base = rows(200, 4), rows(200, 5).mean(axis=0)
    result = calibrate_mean(calibration, rows(200, 6, shift=1.0), base, 20, 99.0, 50, seed=0)
    assert result["check_exceed_rate"] == 1.0


def test_load_mean_baseline(tmp_path) -> None:
    path = tmp_path / "mean.json"
    path.write_text(json.dumps({"metadata": {"created": "t"}, "mean": [0.5, 1.5]}))
    metadata, mean = load_mean_baseline(path)
    assert metadata == {"created": "t"} and mean.tolist() == [0.5, 1.5]
