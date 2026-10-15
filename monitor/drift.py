"""Detect which individual SAE features changed frequency between a baseline and a recent window.

Each feature is tested with a two-sided, two-proportion z-test on raw counts. Because thousands of
features are tested at once, p-values are adjusted with Benjamini-Hochberg, which controls the
false discovery rate: the expected fraction of flagged features that are false alarms.

The whole window also gets one overall score: the Jensen-Shannon divergence between the baseline's
and the window's feature distributions (each feature's share of all top-k slots).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import false_discovery_control, norm

MIN_COUNT = 5


@dataclass(frozen=True)
class FeatureDrift:
    """One feature's change between baseline and window; rates are per 1,000 requests."""

    feature_id: int
    baseline_rate: float
    window_rate: float
    ratio: float
    p_value: float
    adjusted_p: float


def smoothed_rate(count: int, n: int) -> float:
    """Add-one smoothed rate per 1,000: (count + 1) / (n + 2), so a zero count stays above 0."""
    return (count + 1) / (n + 2) * 1000


def two_proportion_p(count_a: int, n_a: int, count_b: int, n_b: int) -> float:
    """Two-sided pooled z-test p-value for count_a/n_a versus count_b/n_b."""
    pooled = (count_a + count_b) / (n_a + n_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (count_a / n_a - count_b / n_b) / se
    return float(2 * norm.sf(abs(z)))


def feature_drift(
    baseline_counts: Mapping[int, int],
    baseline_n: int,
    window_counts: Mapping[int, int],
    window_n: int,
    min_count: int = MIN_COUNT,
) -> list[FeatureDrift]:
    """Test every feature seen at least min_count times on either side; rank by adjusted p."""
    if baseline_n <= 0 or window_n <= 0:
        raise ValueError("baseline_n and window_n must be positive")
    tested = sorted(
        f
        for f in set(baseline_counts) | set(window_counts)
        if max(baseline_counts.get(f, 0), window_counts.get(f, 0)) >= min_count
    )
    if not tested:
        return []
    p_values = [
        two_proportion_p(window_counts.get(f, 0), window_n, baseline_counts.get(f, 0), baseline_n)
        for f in tested
    ]
    adjusted = false_discovery_control(p_values, method="bh")

    results = []
    for f, p, adj in zip(tested, p_values, adjusted, strict=True):
        base_rate = smoothed_rate(baseline_counts.get(f, 0), baseline_n)
        window_rate = smoothed_rate(window_counts.get(f, 0), window_n)
        results.append(
            FeatureDrift(f, base_rate, window_rate, window_rate / base_rate, p, float(adj))
        )
    results.sort(key=lambda r: (r.adjusted_p, -abs(math.log(r.ratio)), r.feature_id))
    return results


def to_distribution(counts: Mapping[int, int], feature_ids: Sequence[int]) -> np.ndarray:
    """Each feature's share of the total count, in feature_ids order; missing features get 0."""
    values = np.array([counts.get(f, 0) for f in feature_ids], dtype=float)
    if (values < 0).any():
        raise ValueError("counts must be non-negative")
    total = values.sum()
    if total <= 0:
        raise ValueError("counts must have a positive total")
    return values / total


def js_divergence(baseline_counts: Mapping[int, int], window_counts: Mapping[int, int]) -> float:
    """Jensen-Shannon divergence in bits: 0 for identical distributions, 1 for disjoint ones."""
    feature_ids = sorted(set(baseline_counts) | set(window_counts))
    p = to_distribution(baseline_counts, feature_ids)
    q = to_distribution(window_counts, feature_ids)
    # scipy returns the Jensen-Shannon distance, the square root of the divergence.
    return float(jensenshannon(p, q, base=2) ** 2)
