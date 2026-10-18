"""Computations behind eval/02_results.ipynb: tables, figures and failure cases, from the cache.

Everything here reads eval/results/experiment_runs.csv, the parquet caches in data/experiments/
and the saved baselines and thresholds; no model is loaded. Extraction overhead is the one
exception: it needs the model running, so it is read from eval/results/latency_<profile>.json,
written by eval/latency.py.
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import chain
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.ticker import FixedLocator, FuncFormatter

from eval.multiseed import NORMAL
from eval.plot_multiseed import GRID, INK, MUTED, PANELS, TITLES
from eval.run_experiment import (
    EMBEDDING,
    EMBEDDINGS_PATH,
    NEAR_LINE,
    RUNS_PATH,
    SAE,
    detector_thresholds,
    features_path,
    ids,
    load_embeddings,
    load_hits,
    replay_scores,
    run_sequences,
)
from monitor.baseline import baseline_path, load_baseline
from monitor.config import load_config
from monitor.drift import feature_drift, score_window
from monitor.model import PROFILE_ENV
from traffic.generate import load_pool

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROFILES = ("gpt2_small", "gemma2_2b")
LABELS = {"gpt2_small": "GPT-2 Small", "gemma2_2b": "Gemma 2 2B"}
# Mean embedding distance reads prompt text only, so its rows are the same under both profiles
# (checked by same_embedding_rows) and it is shown once.
SERIES = (
    (("gpt2_small", SAE), "SAE features, GPT-2 Small", "#2a78d6", "o"),
    (("gpt2_small", EMBEDDING), "Embeddings, mean distance", "#eb6834", "s"),
    (("gemma2_2b", SAE), "SAE features, Gemma 2 2B", "#1baf7a", "^"),
)
SHIFT_POOLS = {
    "medical_ramp": "shift_medical",
    "medical_nowhat_ramp": "shift_medical_nowhat",
    "new_topic_ramp": "shift_new_topic",
    "injection_ramp": "shift_injection",
}
SAMPLE_SEED = 0
# Every shift scenario holds 0% shift until request 500, ramps to 30% by 1,500, then holds.
RAMP = (500, 1500)


@contextmanager
def using_profile(name: str) -> Iterator[None]:
    """Make `name` the active profile inside the block, then restore the previous setting."""
    previous = os.environ.get(PROFILE_ENV)
    os.environ[PROFILE_ENV] = name
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PROFILE_ENV)
        else:
            os.environ[PROFILE_ENV] = previous


def load_runs(path: Path = RUNS_PATH) -> list[dict]:
    """experiment_runs.csv with numbers parsed; empty cells become None."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    numbers = ("first_alert", "delay", "shift_share", "alerts", "alerts_per_1000")
    for row in rows:
        row["seed"] = int(row["seed"])
        for key in (*numbers, "normal_score_ratio"):
            row[key] = float(row[key]) if row.get(key) else None
    return rows


def same_embedding_rows(rows: Sequence[dict]) -> bool:
    """True if the embedding detector's results are identical under every profile."""
    by_profile = {
        p: sorted(
            (r["scenario"], r["seed"], r["first_alert"], r["alerts"])
            for r in rows
            if r["profile"] == p and r["detector"] == EMBEDDING
        )
        for p in PROFILES
    }
    return by_profile[PROFILES[0]] == by_profile[PROFILES[1]]


def spread(values: Sequence[float]) -> dict:
    """Mean, standard deviation, minimum and maximum (None when there are no values)."""
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def results_table(rows: Sequence[dict]) -> list[dict]:
    """One row per series and scenario: delay across seeds, then false alarms on normal_only."""
    table = []
    for (profile, detector), name, _, _ in SERIES:
        mine = [r for r in rows if r["profile"] == profile and r["detector"] == detector]
        normal = [r for r in mine if r["scenario"] == NORMAL]
        false_alarms = spread([r["alerts_per_1000"] for r in normal])
        for scenario in PANELS:
            runs = [r for r in mine if r["scenario"] == scenario]
            delays = [r["delay"] for r in runs if r["delay"] is not None]
            table.append(
                {
                    "series": name,
                    "scenario": scenario,
                    "detected": f"{len(delays)}/{len(runs)}",
                    **{f"delay_{k}": v for k, v in spread(delays).items()},
                    "near_line": sum(
                        r["normal_score_ratio"] is not None and r["normal_score_ratio"] >= NEAR_LINE
                        for r in runs
                    ),
                    **{f"fa_{k}": v for k, v in false_alarms.items()},
                    "fa_runs": f"{sum(r['alerts'] > 0 for r in normal)}/{len(normal)}",
                }
            )
    return table


def head_to_head(rows: Sequence[dict], profile: str) -> dict:
    """Seed-by-seed: in how many runs the SAE detector alerted first, tied, or later or never."""
    index = {(r["detector"], r["scenario"], r["seed"]): r for r in rows if r["profile"] == profile}
    counts = Counter()
    for (detector, scenario, seed), sae in index.items():
        if detector != SAE or scenario == NORMAL:
            continue
        a, b = sae["first_alert"], index[(EMBEDDING, scenario, seed)]["first_alert"]
        if a is not None and (b is None or a < b):
            counts["sae_first"] += 1
        elif a is not None and a == b:
            counts["tie"] += 1
        else:
            counts["embedding_first_or_sae_missed"] += 1
    return dict(counts)


def failure_cases(rows: Sequence[dict]) -> list[dict]:
    """Runs where the SAE detector alerted after the embedding detector, or never."""
    index = {(r["profile"], r["detector"], r["scenario"], r["seed"]): r for r in rows}
    cases = []
    for (profile, detector, scenario, seed), sae in sorted(index.items()):
        if detector != SAE or scenario == NORMAL:
            continue
        emb = index[(profile, EMBEDDING, scenario, seed)]
        if sae["first_alert"] is None or (
            emb["first_alert"] is not None and sae["first_alert"] > emb["first_alert"]
        ):
            cases.append(
                {
                    "profile": profile,
                    "scenario": scenario,
                    "seed": seed,
                    "sae_alert": sae["first_alert"],
                    "embedding_alert": emb["first_alert"],
                    "sae_ratio": sae["normal_score_ratio"],
                    "embedding_ratio": emb["normal_score_ratio"],
                    "embedding_normal_alerted_first": emb["normal_alert_first"] == "True",
                }
            )
    return cases


def replay_one(profile: str, scenario: str, seed: int) -> dict:
    """One run's rows, both detectors' window scores and thresholds, replayed from the cache."""
    window_size = load_config().window_size
    with using_profile(profile):
        rows = run_sequences([seed])[(scenario, seed)]
        thresholds, counts, mean = detector_thresholds(window_size)
        scores = replay_scores(
            {(scenario, seed): rows},
            load_hits(features_path(profile)),
            load_embeddings(EMBEDDINGS_PATH),
            counts,
            mean,
            window_size,
        )
    return {
        "rows": rows,
        "window_size": window_size,
        "thresholds": thresholds,
        "scores": {d: s[(scenario, seed)] for d, s in scores.items()},
    }


def style_axes(ax: plt.Axes) -> None:
    """Recessive grid and axes, as in the other result figures."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def figure_scores(replays: dict[str, dict], scenario: str, seed: int, out: Path) -> None:
    """Figure 1: both detectors' scores over requests (divided by their thresholds), per profile.

    The y axis is logarithmic, so the crossing near 1 stays visible next to scores 10x higher.
    """
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    fig, axes = plt.subplots(
        len(replays), 1, figsize=(10, 3.4 * len(replays)), dpi=150, sharex=True
    )
    for ax, (profile, replay) in zip(np.atleast_1d(axes), replays.items(), strict=True):
        rows, window = replay["rows"], replay["window_size"]
        ax.axvspan(RAMP[0], RAMP[1], color=GRID, alpha=0.45, linewidth=0)
        ax.axvspan(RAMP[1], len(rows), color=GRID, alpha=0.9, linewidth=0)
        for x, text in ((RAMP[0] + 10, "shift ramps 0 → 30%"), (RAMP[1] + 10, "held at 30%")):
            ax.text(x, 0.32, text, fontsize=8, color=MUTED, path_effects=halo)
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.text(window + 5, 1.06, "alarm level", fontsize=8, color=MUTED, path_effects=halo)
        series = [(SAE, s) for s in SERIES if s[0] == (profile, SAE)]
        series += [(EMBEDDING, s) for s in SERIES if s[0][1] == EMBEDDING]
        for detector, (_, name, color, marker) in series:
            ratio = replay["scores"][detector] / replay["thresholds"][detector]
            x = np.arange(window, window + len(ratio))
            ax.plot(x, ratio, color=color, linewidth=2, label=name)
            above = np.flatnonzero(ratio > 1)
            if above.size:
                first = int(above[0])
                ax.plot(x[first], ratio[first], marker=marker, markersize=9, color=color)
                ax.annotate(
                    f"first alert: request {x[first]}",
                    (x[first], ratio[first]),
                    xytext=(-6, 12 if detector == SAE else -18),
                    textcoords="offset points",
                    ha="right",
                    fontsize=8,
                    color=INK,
                    path_effects=halo,
                )
        ax.set_yscale("log")
        ax.set_ylim(0.3, 20)
        ax.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.yaxis.set_minor_locator(FixedLocator([]))
        ax.set_title(LABELS[profile], loc="left", fontsize=10, color=INK)
        ax.set_ylabel("score / threshold (log)", color=MUTED, fontsize=9)
        style_axes(ax)
        ax.legend(loc="upper left", frameon=False, fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("request number", color=MUTED, fontsize=9)
    fig.suptitle(
        f"Drift scores over one run: {TITLES[scenario].lower()}, seed {seed}, window {window}",
        x=0.01,
        ha="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


def figure_delays(rows: Sequence[dict], out: Path) -> None:
    """Figure 2: delay per scenario and series, one dot per seed, with the median as a bar."""
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=150)
    offsets = np.linspace(-0.25, 0.25, len(SERIES))
    jitter = random.Random(SAMPLE_SEED)
    for offset, ((profile, detector), name, color, marker) in zip(offsets, SERIES, strict=True):
        for i, scenario in enumerate(PANELS):
            delays = [
                r["delay"]
                for r in rows
                if (r["profile"], r["detector"], r["scenario"]) == (profile, detector, scenario)
                and r["delay"] is not None
            ]
            x = [i + offset + jitter.uniform(-0.04, 0.04) for _ in delays]
            ax.scatter(
                x,
                delays,
                color=color,
                marker=marker,
                s=22,
                alpha=0.75,
                linewidths=0,
                label=name if i == 0 else None,
            )
            ax.hlines(
                statistics.median(delays),
                i + offset - 0.07,
                i + offset + 0.07,
                color=INK,
                linewidth=2,
            )
    ax.set_xticks(range(len(PANELS)), [TITLES[s] for s in PANELS], fontsize=8, color=INK)
    ax.set_ylabel("requests after shift start\nuntil first alert", color=MUTED, fontsize=9)
    ax.set_ylim(0, None)
    style_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8, ncols=3)
    ax.set_title(
        "Detection delay by scenario at the saved thresholds: one dot per seed, black bar = median",
        loc="left",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def overhead(profile: str) -> dict | None:
    """Median and p95 extraction time and total overhead from latency_<profile>.json, if any."""
    path = RESULTS_DIR / f"latency_{profile}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    on, off = payload["extract_on"]["summary"], payload["extract_off"]["summary"]
    return {
        "profile": profile,
        "checks_passed": payload["checks_passed"],
        "requests": payload["extract_on"]["requests"],
        "extract_ms_p50": on["extract_ms"]["p50"],
        "extract_ms_p95": on["extract_ms"]["p95"],
        "gen_ms_p50": off["gen_ms"]["p50"],
        "total_overhead_pct_p50": payload["total_overhead_pct"].get("p50"),
        "total_overhead_pct_p95": payload["total_overhead_pct"].get("p95"),
    }


def shift_enrichment(
    hits: dict, prompts: Sequence[str], baseline: dict, top: int = 10
) -> list[dict]:
    """Features most over-represented in a shift pool's top-k lists against the baseline.

    These are the features a shift should move. Rates are per 1,000 prompts.
    """
    counts = Counter(chain.from_iterable(ids(hits[p]) for p in prompts))
    n_base, n = baseline["prompt_count"], len(prompts)
    rows = [
        {
            "feature": f,
            "shift_rate": c / n * 1000,
            "baseline_rate": baseline["counts"].get(f, 0) / n_base * 1000,
        }
        for f, c in counts.items()
    ]
    rows.sort(key=lambda r: r["shift_rate"] - r["baseline_rate"], reverse=True)
    return rows[:top]


def window_rates(hits: dict, window: Sequence[dict], features: Sequence[int]) -> dict[int, float]:
    """Per-1,000 rate of each feature in one window's top-k lists."""
    counts = Counter(chain.from_iterable(ids(hits[r["prompt"]]) for r in window))
    return {f: counts.get(f, 0) / len(window) * 1000 for f in features}


def separation(profile: str, scenario: str, n: int = 500) -> dict:
    """How far a window of n shift prompts, and the normal pool, sit from the profile's threshold.

    Both are scored like a drift window and divided by the threshold: shift_ratio says how strong
    a fully shifted window is, normal_ratio how close normal traffic already is.
    """
    window_size = load_config().window_size
    with using_profile(profile):
        hits = load_hits(features_path(profile))
        thresholds, counts, _ = detector_thresholds(window_size)
    rng = random.Random(SAMPLE_SEED)
    pool = shift_prompts(scenario, hits)
    shift = rng.sample(pool, min(n, len(pool)))
    normal = rng.sample(load_pool("normal_test"), n)
    return {
        "shift_ratio": score_window(counts, (ids(hits[p]) for p in shift)) / thresholds[SAE],
        "normal_ratio": score_window(counts, (ids(hits[p]) for p in normal)) / thresholds[SAE],
    }


def shift_prompts(scenario: str, hits: dict) -> list[str]:
    """A scenario's shift prompts that are in the cache: those the replayed seeds send.

    Medical without "What" is derived as in the replay.
    """
    from eval.multiseed import load_pools

    pool = SHIFT_POOLS[scenario]
    return sorted({p for p in load_pools({pool})[pool] if p in hits})


def failure_features(case: dict, top: int = 8) -> dict:
    """For one failure case: shift features and how far they had moved when embeddings alerted.

    Uses the window ending at the embedding detector's first alert, when the SAE detector had not
    alerted yet; also lists what moved most in that window by the per-feature drift test.
    """
    profile, scenario, seed = case["profile"], case["scenario"], case["seed"]
    replay = replay_one(profile, scenario, seed)
    window_size, end = replay["window_size"], int(case["embedding_alert"])
    window = replay["rows"][end - window_size : end]
    with using_profile(profile):
        hits = load_hits(features_path(profile))
        _, baseline = load_baseline(baseline_path())
    enriched = shift_enrichment(hits, shift_prompts(scenario, hits), baseline, top)
    rates = window_rates(hits, window, [e["feature"] for e in enriched])
    window_counts = Counter(chain.from_iterable(ids(hits[r["prompt"]]) for r in window))
    drifted = feature_drift(
        baseline["counts"], baseline["prompt_count"], window_counts, len(window)
    )
    return {
        "shift_share": sum(r["is_shift"] for r in window) / len(window),
        "sae_score_ratio": float(
            replay["scores"][SAE][end - window_size] / replay["thresholds"][SAE]
        ),
        "should_move": [e | {"window_rate": rates[e["feature"]]} for e in enriched],
        "moved_most": [
            {
                "feature": d.feature_id,
                "baseline_rate": d.baseline_rate,
                "window_rate": d.window_rate,
                "adjusted_p": d.adjusted_p,
            }
            for d in drifted[:top]
        ],
    }


def normal_levels(profile: str, quantiles: Sequence[float] = (50, 90, 99)) -> dict[str, dict]:
    """Quantiles of every normal_only window's score / threshold over all seeds, per detector."""
    window_size = load_config().window_size
    with using_profile(profile):
        runs = {k: v for k, v in run_sequences().items() if k[0] == NORMAL}
        thresholds, counts, mean = detector_thresholds(window_size)
        scores = replay_scores(
            runs,
            load_hits(features_path(profile)),
            load_embeddings(EMBEDDINGS_PATH),
            counts,
            mean,
            window_size,
        )
    return {
        detector: {
            q: float(np.percentile(np.concatenate(list(s.values())) / thresholds[detector], q))
            for q in quantiles
        }
        for detector, s in scores.items()
    }


def mixing_curve(
    profile: str,
    scenario: str,
    shares: Sequence[float] = (0.0, 0.02, 0.05, 0.1),
    n_windows: int = 20,
) -> dict[float, float]:
    """Mean SAE score / threshold of random normal_test windows with a share of shift prompts.

    Shows how hard a small share of shifted prompts pushes the score, which is what sets the
    delay early in a ramp.
    """
    window_size = load_config().window_size
    with using_profile(profile):
        hits = load_hits(features_path(profile))
        thresholds, counts, _ = detector_thresholds(window_size)
    normal, shift = load_pool("normal_test"), shift_prompts(scenario, hits)
    rng = random.Random(SAMPLE_SEED)
    curve = {}
    for share in shares:
        k = round(share * window_size)
        ratios = [
            score_window(
                counts,
                (ids(hits[p]) for p in rng.sample(normal, window_size - k) + rng.sample(shift, k)),
            )
            / thresholds[SAE]
            for _ in range(n_windows)
        ]
        curve[share] = statistics.mean(ratios)
    return curve
