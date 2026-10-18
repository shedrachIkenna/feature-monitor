"""Plot detection delay against false alarms for both profiles, one panel per shift scenario.

Reads eval/results/experiment_sweep.csv (eval.run_experiment replay). Each point is one threshold
of the sweep: x is the share of normal_only windows above it, y the median delay over the seeds,
and the bar spans the fastest and slowest seed. Rings mark the saved (99th percentile)
thresholds. Mean embedding distance does not depend on the profile, so it is drawn once. The
table view of every point is the CSV itself.

Run with: uv run python -m eval.plot_experiment
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from eval.plot_multiseed import GRID, INK, MUTED, PANELS, TITLES

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SWEEP_PATH = RESULTS_DIR / "experiment_sweep.csv"
OUT_PATH = RESULTS_DIR / "experiment_tradeoff.png"
SAVED_PERCENTILE = 99.0
# Categorical slots 1-3 of the dataviz reference palette (validator: all checks pass; the aqua
# is below 3:1 contrast, so markers, the legend and the CSV carry identity too).
SERIES = (
    (("gpt2_small", "sae"), "SAE features, GPT-2 Small", "#2a78d6", "o"),
    (("gpt2_small", "embedding_mean"), "Embeddings, mean distance", "#eb6834", "s"),
    (("gemma2_2b", "sae"), "SAE features, Gemma 2 2B", "#1baf7a", "^"),
)


def load_sweep(path: Path = SWEEP_PATH) -> list[dict]:
    """Rows of the sweep CSV with numbers parsed; empty delays become None."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("percentile", "fa_window_rate"):
            row[key] = float(row[key])
        for key in ("median_delay", "min_delay", "max_delay"):
            row[key] = float(row[key]) if row[key] else None
    return rows


def plot_tradeoff(rows: list[dict], out: Path) -> None:
    """Save the four-panel delay vs false-alarm figure."""
    slowest = max(r["max_delay"] for r in rows if r["max_delay"] is not None)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.8), dpi=150, sharex=True, sharey=True)
    for ax, scenario in zip(axes.flat, PANELS, strict=True):
        for (profile, detector), name, color, marker in SERIES:
            points = sorted(
                (
                    r
                    for r in rows
                    if (r["profile"], r["detector"], r["scenario"]) == (profile, detector, scenario)
                    and r["median_delay"] is not None
                ),
                key=lambda r: r["percentile"],
            )
            if not points:
                continue
            x = [100 * r["fa_window_rate"] for r in points]
            y = [r["median_delay"] for r in points]
            low = [m - r["min_delay"] for r, m in zip(points, y, strict=True)]
            high = [r["max_delay"] - m for r, m in zip(points, y, strict=True)]
            bars = ax.errorbar(
                x,
                y,
                yerr=[low, high],
                color=color,
                marker=marker,
                markersize=5,
                linewidth=2,
                elinewidth=1,
                capsize=0,
                label=name,
            )
            bars[2][0].set_alpha(0.35)
            saved = next(r for r in points if r["percentile"] == SAVED_PERCENTILE)
            ax.plot(
                100 * saved["fa_window_rate"],
                saved["median_delay"],
                marker="o",
                markersize=13,
                markerfacecolor="none",
                markeredgecolor=color,
                markeredgewidth=1.2,
            )
        ax.set_title(TITLES[scenario], loc="left", fontsize=10, color=INK)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(-0.01, 25)
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
    fig.legend(handles, labels, loc="lower center", frameon=False, fontsize=9, ncols=3)
    fig.suptitle(
        "Detection delay vs false alarms, GPT-2 Small and Gemma 2 2B, 10 seeds per scenario\n"
        "Lower-left is better. Median over seeds; bars: fastest to slowest seed; "
        "rings: saved 99th-percentile thresholds",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    """Plot the sweep of both profiles to eval/results/experiment_tradeoff.png."""
    plot_tradeoff(load_sweep(), OUT_PATH)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
