import math

import numpy as np

from monitor.drift import feature_drift, smoothed_rate

N = 3000
N_FEATURES = 2000
ALPHA = 0.05
INJECTED = 7


def true_rates(rng: np.random.Generator) -> np.ndarray:
    """Per-request rates spread log-uniformly from 0.03% to 20%, like the real baseline's range."""
    rates = np.exp(rng.uniform(math.log(0.0003), math.log(0.2), N_FEATURES))
    rates[INJECTED] = 0.01
    return rates


def draw_counts(rng: np.random.Generator, rates: np.ndarray, n: int) -> dict[int, int]:
    counts = rng.binomial(n, rates)
    return {f: int(c) for f, c in enumerate(counts) if c > 0}


def test_injected_10x_feature_is_found() -> None:
    rng = np.random.default_rng(0)
    rates = true_rates(rng)
    baseline = draw_counts(rng, rates, N)
    shifted = rates.copy()
    shifted[INJECTED] *= 10
    window = draw_counts(rng, shifted, N)

    results = feature_drift(baseline, N, window, N)

    assert results[0].feature_id == INJECTED
    assert results[0].adjusted_p < ALPHA
    assert results[0].ratio > 5


def test_noise_of_baseline_size_flags_nothing() -> None:
    rng = np.random.default_rng(1)
    rates = true_rates(rng)
    baseline = draw_counts(rng, rates, N)
    window = draw_counts(rng, rates, N)

    results = feature_drift(baseline, N, window, N)

    assert len(results) > 1000
    assert [r.feature_id for r in results if r.adjusted_p < ALPHA] == []


def test_zero_counts_give_finite_rates_and_ratio() -> None:
    results = feature_drift({1: 50}, 1000, {}, 1000)

    assert results[0].window_rate == smoothed_rate(0, 1000) > 0
    assert math.isfinite(results[0].ratio)
    assert results[0].adjusted_p < ALPHA


def test_features_below_min_count_on_both_sides_are_skipped() -> None:
    results = feature_drift({1: 4, 2: 5}, 1000, {1: 4, 3: 6}, 1000, min_count=5)

    assert sorted(r.feature_id for r in results) == [2, 3]
