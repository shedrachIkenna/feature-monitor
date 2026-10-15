import pytest

from monitor.baseline import build_baseline, load_baseline, sample_prompts, save_baseline
from monitor.extract import FeatureHit

FAKE_HITS = {
    "a": [1, 2, 3],
    "b": [2, 3],
    "c": [3],
    "d": [3, 7],
}


def fake_extract(prompt: str, k: int) -> list[FeatureHit]:
    return [FeatureHit(feature_id=f, activation=1.0) for f in FAKE_HITS[prompt][:k]]


def test_build_baseline_counts_and_rates() -> None:
    baseline = build_baseline(["a", "b", "c", "d"], extract=fake_extract)

    assert baseline["prompt_count"] == 4
    assert baseline["counts"] == {1: 1, 2: 2, 3: 4, 7: 1}
    assert baseline["rate_per_1000"] == {1: 250.0, 2: 500.0, 3: 1000.0, 7: 250.0}


def test_build_baseline_respects_k() -> None:
    baseline = build_baseline(["a", "d"], extract=fake_extract, k=1)

    assert baseline["counts"] == {1: 1, 3: 1}


def test_build_baseline_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_baseline([], extract=fake_extract)


def test_sample_prompts_is_seeded_and_distinct() -> None:
    pool = [f"p{i}" for i in range(100)]

    first = sample_prompts(pool, 30, seed=0)

    assert first == sample_prompts(pool, 30, seed=0)
    assert first != sample_prompts(pool, 30, seed=1)
    assert len(set(first)) == 30


def test_sample_prompts_rejects_n_larger_than_pool() -> None:
    with pytest.raises(ValueError):
        sample_prompts(["x", "y"], 3)


def test_save_then_load_round_trips(tmp_path) -> None:
    baseline = build_baseline(["a", "b", "c"], extract=fake_extract)
    metadata = {"model": "gpt2", "layer": 8, "seed": 0}
    path = tmp_path / "nested" / "gpt2_8.json"

    save_baseline(baseline, metadata, path)
    loaded_metadata, loaded = load_baseline(path)

    assert loaded_metadata == metadata
    assert loaded == baseline
