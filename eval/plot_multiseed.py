"""Plot detection delay against false-alarm rate for both detectors, one panel per shift scenario.

Reads eval/results/multiseed.json written by eval.multiseed. Each point is one threshold from the
sweep: x is the share of normal_only windows above it, y the median delay over the seeds, and the
bar spans the fastest and slowest seed. The ringed points are the saved (99th percentile) thresholds.

Run with: uv run python -m eval.plot_multiseed
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

RESULTS_DIR = Path(__file__).resolve().parent / "results"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e1e0d9"
STYLE = {
    "sae": {"name": "SAE features", "color": "#2b2b29", "marker": "o"},
    "embedding": {"name": "Sentence embeddings", "color": "#8a8882", "marker": "s"},
}
PANELS = ("medical_ramp", "medical_nowhat_ramp", "new_topic_ramp", "injection_ramp")
TITLES = {
    "medical_ramp": "Medical questions",
    "medical_nowhat_ramp": 'Medical, without "What…" openers',
    "new_topic_ramp": "New topic, same style (ACCOUNT)",
    "injection_ramp": "Prompt injections",
}


def plot_tradeoff(payload: dict, out: Path) -> None:
    """Save the four-panel delay vs false-alarm figure."""
    seeds = len(payload["seeds"])
    slowest = max(
        p[scenario]["max"]
        for r in payload["detectors"].values()
        for p in r["sweep"]
        for scenario in PANELS
        if p[scenario]["max"] is not None
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), dpi=150, sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, PANELS, strict=True):
        for detector, style in STYLE.items():
            result = payload["detectors"][detector]
            points = [p for p in result["sweep"] if p[scenario]["median"] is not None]
            x = [100 * p["window_rate"] for p in points]
            y = [p[scenario]["median"] for p in points]
            low = [m - p[scenario]["min"] for p, m in zip(points, y, strict=True)]
            high = [p[scenario]["max"] - m for p, m in zip(points, y, strict=True)]
            ax.errorbar(
                x,
                y,
                yerr=[low, high],
                color=style["color"],
                marker=style["marker"],
                markersize=5,
                linewidth=2,
                elinewidth=1,
                capsize=0,
                label=style["name"],
            )
            saved = next(p for p in points if p["threshold"] == result["saved_threshold"])
            ax.plot(
                100 * saved["window_rate"],
                saved[scenario]["median"],
                marker="o",
                markersize=13,
                markerfacecolor="none",
                markeredgecolor=style["color"],
                markeredgewidth=1.2,
            )
        ax.set_title(TITLES[scenario], loc="left", fontsize=10, color=INK)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(-0.01, 15)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.1, 1, 10]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.set_ylim(0, slowest * 1.05)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(
            "normal windows above threshold (log scale above 0.1%)", color=MUTED, fontsize=9
        )
    for ax in axes[:, 0]:
        ax.set_ylabel("requests after shift start\nuntil first alert", color=MUTED, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncols=2)
    fig.suptitle(
        f"Detection delay vs false alarms, {seeds} seeds per scenario, window "
        f"{payload['config']['window_size']}\n"
        "Lower-left is better. Median over seeds; bars: fastest to slowest seed; "
        "rings: saved 99th-percentile thresholds",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """Plot eval/results/multiseed.json to eval/results/multiseed_tradeoff.png."""
    payload = json.loads((RESULTS_DIR / "multiseed.json").read_text(encoding="utf-8"))
    out = RESULTS_DIR / "multiseed_tradeoff.png"
    plot_tradeoff(payload, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
