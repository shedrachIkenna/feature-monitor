from collections import Counter

from eval.sanity import compare_groups, feature_counts


def test_feature_counts_counts_each_prompt_once() -> None:
    assert feature_counts([[1, 2, 2], [2, 3]]) == Counter({1: 1, 2: 2, 3: 1})


def test_compare_groups_finds_enriched_features_and_jaccard() -> None:
    counts_a = Counter({10: 5, 11: 7, 12: 6})
    counts_b = Counter({10: 1, 11: 2, 12: 5, 20: 9})

    result = compare_groups(counts_a, counts_b)

    assert result.common_a == {10, 11, 12}
    assert result.common_b == {12, 20}
    assert result.enriched_a == {10}
    assert result.enriched_b == {20}
    assert result.jaccard == 1 / 4
    assert result.verdict == "weakly"


def test_compare_groups_clear_and_none_verdicts() -> None:
    clear = compare_groups(Counter({1: 5, 2: 5, 3: 5}), Counter({7: 5, 8: 5, 9: 5}))
    none = compare_groups(Counter({1: 5}), Counter({1: 5}))

    assert clear.verdict == "clearly"
    assert clear.jaccard == 0.0
    assert none.verdict == "not at all"
    assert none.jaccard == 1.0
