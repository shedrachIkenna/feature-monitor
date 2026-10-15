from pathlib import Path

from eval.plot_normal_jsd import plot_scores


def test_plot_scores_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "scores.png"

    plot_scores([0.01, 0.02, 0.02, 0.03], 0.025, "test", out)

    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
