import json

from eval.recorded_example import OUT, RESULTS_DIR, build, first_above


def test_committed_example_matches_the_results_files() -> None:
    multiseed = json.loads((RESULTS_DIR / "multiseed.json").read_text())
    embed = json.loads((RESULTS_DIR / "embed_baselines.json").read_text())
    assert json.loads(OUT.read_text()) == build(multiseed, embed)


def test_lines_share_their_request_numbers() -> None:
    example = json.loads(OUT.read_text())
    n = len(example["request_numbers"])
    assert len(example["sae_shift"]) == len(example["sae_no_shift"]) == n
    assert example["request_numbers"][0] == example["window_size"]


def test_first_above_counts_from_the_first_window() -> None:
    assert first_above([0.5, 1.0, 1.2, 0.9], 500) == 502
    assert first_above([0.5, 0.99], 500) is None
