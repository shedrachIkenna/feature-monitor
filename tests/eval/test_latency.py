import pytest

from eval.latency import load_prompts, overhead_pct, percentile, summarise


def test_prompt_file_has_100_distinct_prompts() -> None:
    prompts = load_prompts()
    assert len(prompts) == 100
    assert len(set(prompts)) == 100


@pytest.mark.parametrize(("q", "expected"), [(0, 1.0), (50, 2.5), (95, 3.85), (100, 4.0)])
def test_percentile_interpolates_linearly(q, expected) -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(expected)


def test_percentile_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        percentile([], 50)


def test_summarise_reports_p50_and_p95_per_metric() -> None:
    rows = [{"gen_ms": float(i), "extract_ms": 0.0, "total_ms": 2.0 * i} for i in range(101)]
    summary = summarise(rows)
    assert summary["gen_ms"] == {"p50": 50.0, "p95": 95.0}
    assert summary["extract_ms"] == {"p50": 0.0, "p95": 0.0}
    assert summary["total_ms"] == {"p50": 100.0, "p95": 190.0}


def test_overhead_is_relative_to_the_run_without_extraction() -> None:
    assert overhead_pct(110.0, 100.0) == pytest.approx(10.0)
