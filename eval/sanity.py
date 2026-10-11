"""Topic-separation statistics for the sanity-check notebook."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

COMMON_MIN_PROMPTS = 5
RARE_MAX_PROMPTS = 1
CLEAR_MIN_ENRICHED = 3
CLEAR_MAX_JACCARD = 0.25


@dataclass(frozen=True)
class Separation:
    """How well two prompt groups separate by their top features."""

    common_a: frozenset[int]
    common_b: frozenset[int]
    enriched_a: frozenset[int]
    enriched_b: frozenset[int]
    jaccard: float
    verdict: str


def feature_counts(prompt_features: Iterable[Iterable[int]]) -> Counter[int]:
    """Count the number of prompts each feature appears in."""
    return Counter(feature for features in prompt_features for feature in set(features))


def compare_groups(counts_a: Counter[int], counts_b: Counter[int]) -> Separation:
    """Apply the pre-set common, rare, enrichment and verdict rule to two groups."""
    common_a = frozenset(f for f, n in counts_a.items() if n >= COMMON_MIN_PROMPTS)
    common_b = frozenset(f for f, n in counts_b.items() if n >= COMMON_MIN_PROMPTS)
    enriched_a = frozenset(f for f in common_a if counts_b[f] <= RARE_MAX_PROMPTS)
    enriched_b = frozenset(f for f in common_b if counts_a[f] <= RARE_MAX_PROMPTS)
    union = common_a | common_b
    jaccard = len(common_a & common_b) / len(union) if union else 0.0

    if (
        len(enriched_a) >= CLEAR_MIN_ENRICHED
        and len(enriched_b) >= CLEAR_MIN_ENRICHED
        and jaccard <= CLEAR_MAX_JACCARD
    ):
        verdict = "clearly"
    elif enriched_a or enriched_b:
        verdict = "weakly"
    else:
        verdict = "not at all"
    return Separation(common_a, common_b, enriched_a, enriched_b, jaccard, verdict)
