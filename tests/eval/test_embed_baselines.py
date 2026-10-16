import numpy as np
import pytest

from eval.embed_baselines import (
    MMD,
    calibration_windows,
    embeddings_for,
    gaussian_kernel,
    median_width,
    rolling_mean_distance,
)
from monitor.calibrate import window_scores
from monitor.drift import score_window
from monitor.embed_mean import mean_distance


def unit_rows(n: int, dim: int, seed: int) -> np.ndarray:
    rows = np.random.default_rng(seed).normal(size=(n, dim))
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_rolling_mean_distance_matches_direct_score() -> None:
    data, base = unit_rows(40, 6, 0), unit_rows(30, 6, 1).mean(axis=0)
    rolling = rolling_mean_distance(data, base, 10)
    assert len(rolling) == 31
    for i, score in enumerate(rolling):
        assert score == pytest.approx(mean_distance(data[i : i + 10], base), abs=1e-12)


def test_mean_distance_is_zero_for_the_baseline_itself() -> None:
    base = unit_rows(20, 4, 2)
    assert mean_distance(base, base.mean(axis=0)) == pytest.approx(0.0, abs=1e-15)


def test_mmd_rolling_matches_direct_score() -> None:
    data, base = unit_rows(35, 5, 3), unit_rows(25, 5, 4)
    mmd = MMD(base, width=0.8)
    rolling = mmd.rolling(data, 12)
    assert len(rolling) == 24
    for i, score in enumerate(rolling):
        assert score == pytest.approx(mmd.score(data[i : i + 12]), abs=1e-12)


def test_mmd_is_near_zero_for_the_same_distribution_and_larger_for_a_shift() -> None:
    base = unit_rows(300, 5, 5)
    mmd = MMD(base, width=median_width(base, seed=0))
    same = mmd.score(unit_rows(300, 5, 6))
    shifted = unit_rows(300, 5, 7) + np.array([1.0, 0, 0, 0, 0])
    assert same < 0.01 < mmd.score(shifted)


def test_gaussian_kernel_is_one_on_identical_rows_and_symmetric() -> None:
    a = unit_rows(4, 3, 8)
    k = gaussian_kernel(a, a, 1.0)
    assert np.allclose(np.diag(k), 1.0)
    assert np.allclose(k, k.T)


def test_calibration_windows_are_the_ones_monitor_calibrate_draws() -> None:
    sets = [[i % 7] for i in range(60)]
    counts = {i: 5 for i in range(7)}
    expected = window_scores(sets, np.arange(60), counts, 10, 5, np.random.default_rng([3, 10]))
    windows = calibration_windows(60, 10, 5, seed=3)
    assert [score_window(counts, [sets[i] for i in w]) for w in windows] == list(expected)


def test_embeddings_for_embeds_only_uncached_prompts(tmp_path, monkeypatch) -> None:
    import monitor.embed_drift

    calls = []

    def fake_embed(prompts):
        calls.append(list(prompts))
        return np.array([[float(len(p)), 0.0] for p in prompts], dtype=np.float32)

    monkeypatch.setattr(monitor.embed_drift, "embed_prompts", fake_embed)
    cache = tmp_path / "e.npz"

    embeddings_for(["a", "bb"], cache)
    got = embeddings_for(["bb", "ccc"], cache)

    assert calls == [["a", "bb"], ["ccc"]]
    assert got["ccc"][0] == 3.0 and got["bb"][0] == 2.0
