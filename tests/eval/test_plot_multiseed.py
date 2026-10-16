from eval.plot_multiseed import PANELS, plot_tradeoff


def point(threshold: float, rate: float, median: float) -> dict:
    summary = {"detected": 2, "runs": 2, "median": median, "min": median - 1, "max": median + 1}
    return {"threshold": threshold, "window_rate": rate, **{s: summary for s in PANELS}}


def test_plot_tradeoff_writes_a_png(tmp_path) -> None:
    detector = {"saved_threshold": 0.2, "sweep": [point(0.1, 0.05, 50), point(0.2, 0.0, 80)]}
    payload = {
        "seeds": [0, 1],
        "config": {"window_size": 500},
        "detectors": {"sae": detector, "embedding": detector},
    }
    out = tmp_path / "tradeoff.png"

    plot_tradeoff(payload, out)

    assert out.read_bytes().startswith(b"\x89PNG")
