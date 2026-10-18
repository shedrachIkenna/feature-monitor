import pytest

from monitor import prefetch
from monitor.baseline import baseline_path


def test_baseline_feature_ids_are_sorted_distinct_ints() -> None:
    if not baseline_path().exists():
        pytest.skip(f"no saved baseline for this profile yet: {baseline_path().name}")
    ids = prefetch.baseline_feature_ids()

    assert ids
    assert ids == sorted(set(ids))
    assert all(isinstance(i, int) for i in ids)


def test_main_labels_step_prefetches_baseline_features(monkeypatch, capsys) -> None:
    received: list[list[int]] = []
    monkeypatch.setattr(prefetch, "baseline_feature_ids", lambda: [4, 5])
    monkeypatch.setattr(prefetch, "load_model_and_sae", lambda: pytest.fail("loaded models"))
    monkeypatch.setattr(prefetch, "prefetch_labels", lambda ids: received.append(ids) or (1, 1, 0))

    prefetch.main(["labels"])

    assert received == [[4, 5]]
    assert "1 already cached, 1 fetched, 0 failed" in capsys.readouterr().out
