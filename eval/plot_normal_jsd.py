"""Plot the distribution of normal-window drift scores with the calibrated threshold.

Reads eval/results/normal_jsd.json written by monitor.calibrate.

Run with: uv run python -m eval.plot_normal_jsd
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BAR = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"


def plot_scores(scores: list[float], threshold: float, title: str, out: Path) -> None:
    """Save a histogram of window scores with the threshold as a dashed vertical line."""
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    ax.hist(scores, bins=40, color=BAR, edgecolor="white", linewidth=1)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.5)
    ax.annotate(
        f"threshold {threshold:.4f}",
        xy=(threshold, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        color=INK,
        fontsize=9,
    )
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Jensen-Shannon divergence from baseline (bits)", color=MUTED)
    ax.set_ylabel("windows", color=MUTED)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """Plot the configured window size's calibration-half scores."""
    payload = json.loads((RESULTS_DIR / "normal_jsd.json").read_text(encoding="utf-8"))
    config = payload["config"]
    result = next(r for r in payload["results"] if r["window_size"] == config["window_size"])
    title = (
        f"Normal windows of {result['window_size']} requests "
        f"({len(result['calibration_scores'])} windows from unused normal_train)\n"
        f"normal_test windows exceed the threshold in {result['check_exceed_rate']:.1%} of cases"
    )
    out = RESULTS_DIR / "normal_jsd.png"
    plot_scores(result["calibration_scores"], result["threshold"], title, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
