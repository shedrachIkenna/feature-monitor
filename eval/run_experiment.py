"""Offline experiment harness: encode prompts once per profile, then calibrate and replay from disk.

Steps, run in the order given (FEATURE_MONITOR_PROFILE picks the model):
- extract: encode every prompt any run can use (all of normal_train and normal_test, plus the
  shift prompts that SEEDS draw) into data/experiments/<profile>/features.parquet (top-k SAE
  features and activations), and their MiniLM embeddings into data/experiments/embeddings.parquet,
  shared by all profiles because the embedder reads prompt text only. Only uncached prompts are
  encoded, in chunks saved as they finish, so a stopped run resumes where it stopped.
- calibrate: build the profile's SAE baseline (monitor.baseline's sample) and threshold
  (monitor.calibrate's pools, seed and draw order) from the cache. Files that already exist must
  be reproduced exactly, or the step stops.
- replay: for every scenario and seed, rebuild the request sequence with the traffic generator,
  look up each prompt's cached representation, score every window with both detectors (SAE
  features, and mean embedding distance, the live embedding detector) and write this profile's rows
  of eval/results/experiment_runs.csv (one row per run) and experiment_alerts.csv (one per alert).
- verify: replay, then compare every window's score with eval/results/multiseed.json (SAE) and
  embed_baselines.json (mean distance), which were made before this harness; the first alert of
  every run must be the same.

Why a cache: a prompt's features do not depend on which scenario, seed or detector uses it, so
the model runs once per prompt, and every replay afterwards only counts. It is also fairer: every
detector and seed sees exactly the same representation of the same prompt, with nothing from
timing, server load or the scheduler, so differences between results come from the detectors.

An alert is a window scoring above the threshold right after one at or below it (the first full
window counts if it is above), so a shift that stays above the line is one alert, not hundreds.
Delay and shift share are defined as in eval/multiseed.py.

Run with: uv run python -m eval.run_experiment extract calibrate replay
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eval.embed_baselines import rolling_mean_distance
from eval.multiseed import (
    NORMAL,
    SEEDS,
    delay_summary,
    detection,
    first_alert,
    load_pools,
    load_scenarios,
    mark_normal_first,
    rolling_scores,
    sequences,
    shift_share,
)
from monitor.baseline import (
    N_PROMPTS,
    POOL,
    POOL_DIR,
    SEED,
    K,
    baseline_metadata,
    baseline_path,
    build_baseline,
    load_baseline,
    read_pool,
    sample_prompts,
    save_baseline,
)
from monitor.calibrate import (
    CALIBRATION_POOL,
    CHECK_POOL,
    calibrate,
    load_threshold,
    pool_prompts,
    save_threshold,
    threshold_path,
)
from monitor.config import DriftConfig, load_config
from monitor.embed_mean import load_mean_baseline, mean_baseline_path
from monitor.extract import FeatureHit
from monitor.model import active_profile

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "data" / "experiments"
EMBEDDINGS_PATH = EXPERIMENTS_DIR / "embeddings.parquet"
RESULTS_DIR = ROOT / "eval" / "results"
RUNS_PATH = RESULTS_DIR / "experiment_runs.csv"
ALERTS_PATH = RESULTS_DIR / "experiment_alerts.csv"
MULTISEED_PATH = RESULTS_DIR / "multiseed.json"
EMBED_BASELINES_PATH = RESULTS_DIR / "embed_baselines.json"
CHUNK = 500
SAE = "sae"
EMBEDDING = "embedding_mean"
DETECTORS = (SAE, EMBEDDING)
# Saved scores were rounded (6 decimals; 8 significant digits), and re-embedding in other batches
# moves the last float32 bits, so only these small gaps are tolerated.
VERIFY_TOLERANCE = {SAE: 1e-6, EMBEDDING: 1e-6}

FEATURE_SCHEMA = pa.schema(
    [
        ("prompt", pa.string()),
        ("feature_ids", pa.list_(pa.int32())),
        ("activations", pa.list_(pa.float32())),
    ]
)
EMBEDDING_SCHEMA = pa.schema([("prompt", pa.string()), ("embedding", pa.list_(pa.float32()))])
RUN_FIELDS = [
    "profile",
    "detector",
    "scenario",
    "seed",
    "threshold",
    "window_size",
    "requests",
    "first_alert",
    "delay",
    "shift_share",
    "missed",
    "alerts",
    "alerts_per_1000",
    "normal_alert_first",
]
ALERT_FIELDS = [
    "profile",
    "detector",
    "scenario",
    "seed",
    "request_number",
    "score",
    "threshold",
    "shift_share",
]

Encoder = Callable[[list[str]], dict[str, list]]
Runs = dict[tuple[str, int], list[dict]]
Scores = dict[str, dict[tuple[str, int], np.ndarray]]


def features_path(profile: str) -> Path:
    """Return data/experiments/<profile>/features.parquet."""
    return EXPERIMENTS_DIR / profile / "features.parquet"


def run_sequences(seeds: Sequence[int] = SEEDS) -> Runs:
    """The exact prompt sequence of every scenario and seed."""
    scenarios = load_scenarios()
    names = {s.normal_pool for s in scenarios.values()}
    names |= {s.shift_pool for s in scenarios.values() if s.shift_pool}
    return sequences(scenarios, seeds, load_pools(names))


def needed_prompts(runs: Runs) -> list[str]:
    """Every prompt the baseline, calibration, check pool or any run can use, sorted."""
    prompts = set(read_pool(POOL_DIR / f"{POOL}.jsonl"))
    prompts |= set(read_pool(POOL_DIR / f"{CHECK_POOL}.jsonl"))
    prompts |= {row["prompt"] for rows in runs.values() for row in rows}
    return sorted(prompts)


def extend_cache(
    path: Path, prompts: Sequence[str], encode: Encoder, schema: pa.Schema, chunk: int = CHUNK
) -> int:
    """Encode the prompts a parquet cache lacks, saving after every chunk; returns how many."""
    table = pq.read_table(path, schema=schema) if path.exists() else schema.empty_table()
    missing = sorted(set(prompts) - set(table.column("prompt").to_pylist()))
    start = time.perf_counter()
    for done in range(0, len(missing), chunk):
        batch = missing[done : done + chunk]
        table = pa.concat_tables([table, pa.table({"prompt": batch, **encode(batch)}, schema)])
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        pq.write_table(table, partial)
        os.replace(partial, path)  # a stop mid-write leaves the previous file intact
        finished = done + len(batch)
        each = (time.perf_counter() - start) / finished
        print(
            f"{path.name}: {finished}/{len(missing)} new prompts, {each * 1000:.0f} ms each,"
            f" about {each * (len(missing) - finished) / 60:.0f} min left",
            flush=True,
        )
    return len(missing)


def encode_features(prompts: list[str]) -> dict[str, list]:
    """Top-k SAE features of each prompt with the active profile's model."""
    from monitor.extract import extract_features

    hits = [extract_features(p, K) for p in prompts]
    return {
        "feature_ids": [[h.feature_id for h in row] for row in hits],
        "activations": [[h.activation for h in row] for row in hits],
    }


def encode_embeddings(prompts: list[str]) -> dict[str, list]:
    """Unit-length MiniLM embedding of each prompt."""
    from monitor.embed_drift import embed_prompts

    return {"embedding": embed_prompts(prompts).tolist()}


def load_hits(path: Path) -> dict[str, list[FeatureHit]]:
    """Each cached prompt's top-k features, strongest first."""
    table = pq.read_table(path, schema=FEATURE_SCHEMA).to_pydict()
    return {
        prompt: [FeatureHit(f, a) for f, a in zip(ids, acts, strict=True)]
        for prompt, ids, acts in zip(
            table["prompt"], table["feature_ids"], table["activations"], strict=True
        )
    }


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    """Each cached prompt's embedding as float64, the precision the mean detector scores in."""
    table = pq.read_table(path, schema=EMBEDDING_SCHEMA)
    prompts = table.column("prompt").to_pylist()
    flat = np.asarray(table.column("embedding").combine_chunks().flatten(), dtype=float)
    return dict(zip(prompts, flat.reshape(len(prompts), -1), strict=True))


def ids(hits: Sequence[FeatureHit]) -> list[int]:
    """Feature IDs of a prompt's hits, in order."""
    return [h.feature_id for h in hits]


def calibrate_profile(hits: dict[str, list[FeatureHit]], config: DriftConfig) -> dict:
    """Build (or reproduce) the profile's baseline and threshold from cached features."""
    sample = sample_prompts(read_pool(POOL_DIR / f"{POOL}.jsonl"), N_PROMPTS, SEED)
    baseline = build_baseline(sample, extract=lambda prompt, k: hits[prompt][:k])
    path = baseline_path()
    if path.exists():
        metadata, saved = load_baseline(path)
        if saved != baseline:
            raise RuntimeError(f"cached features do not reproduce {path.name}")
    else:
        metadata = baseline_metadata(baseline["prompt_count"])
        save_baseline(baseline, metadata, path)

    sets = {
        name: [ids(hits[p]) for p in prompts] for name, prompts in pool_prompts(metadata).items()
    }
    result = calibrate(
        sets[CALIBRATION_POOL],
        sets[CHECK_POOL],
        baseline["counts"],
        config.window_size,
        config.threshold_percentile,
        config.calibration_windows,
        config.seed,
    )
    t_path = threshold_path(path)
    if t_path.exists():
        if load_threshold(t_path, config.window_size, metadata["created"]) != result["threshold"]:
            raise RuntimeError(f"cached features do not reproduce {t_path.name}")
    else:
        sizes = {name: len(s) for name, s in sets.items()}
        save_threshold(result, config, metadata, sizes, t_path)
    return {
        "baseline": path.name,
        "features_in_baseline": len(baseline["counts"]),
        "threshold": result["threshold"],
        "check_exceed_rate": result["check_exceed_rate"],
    }


def detector_thresholds(window_size: int) -> tuple[dict[str, float], dict[int, int], np.ndarray]:
    """Saved thresholds of both detectors, the SAE baseline counts and the baseline mean."""
    metadata, baseline = load_baseline(baseline_path())
    mean_metadata, mean = load_mean_baseline(mean_baseline_path())
    thresholds = {
        SAE: load_threshold(threshold_path(baseline_path()), window_size, metadata["created"]),
        EMBEDDING: load_threshold(
            threshold_path(mean_baseline_path()), window_size, mean_metadata["created"]
        ),
    }
    return thresholds, baseline["counts"], mean


def replay_scores(
    runs: Runs,
    hits: dict[str, list[FeatureHit]],
    embeddings: dict[str, np.ndarray],
    counts: dict[int, int],
    mean: np.ndarray,
    window_size: int,
) -> Scores:
    """Every full window's score in every run, per detector."""
    return {
        SAE: {
            key: rolling_scores([ids(hits[r["prompt"]]) for r in rows], counts, window_size)
            for key, rows in runs.items()
        },
        EMBEDDING: {
            key: rolling_mean_distance(
                np.array([embeddings[r["prompt"]] for r in rows]), mean, window_size
            )
            for key, rows in runs.items()
        },
    }


def alert_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Window indices where the score rises above threshold (the first window counts if above)."""
    above = scores > threshold
    return np.flatnonzero(above & ~np.concatenate([[False], above[:-1]]))


def tabulate(
    profile: str, runs: Runs, scores: Scores, thresholds: dict[str, float], window_size: int
) -> tuple[list[dict], list[dict]]:
    """One row per run and one row per alert, for every detector."""
    run_rows, alert_rows = [], []
    for detector in DETECTORS:
        threshold = thresholds[detector]
        for (name, seed), s in scores[detector].items():
            rows = runs[(name, seed)]
            alerts = alert_indices(s, threshold) + window_size
            row = {
                "profile": profile,
                "detector": detector,
                "scenario": name,
                "seed": seed,
                "threshold": threshold,
                "window_size": window_size,
                "requests": len(rows),
                "first_alert": int(alerts[0]) if alerts.size else None,
                "alerts": int(alerts.size),
                "alerts_per_1000": alerts.size / len(rows) * 1000,
            }
            if name != NORMAL:
                normal_first = first_alert(scores[detector][(NORMAL, seed)], threshold, window_size)
                found = mark_normal_first(detection(rows, s, threshold, window_size), normal_first)
                row |= {
                    "delay": found["delay"],
                    "shift_share": found["shift_share"],
                    "missed": found["first_alert"] is None,
                    "normal_alert_first": found["normal_alert_first"],
                }
            run_rows.append(row)
            alert_rows += [
                {
                    "profile": profile,
                    "detector": detector,
                    "scenario": name,
                    "seed": seed,
                    "request_number": int(n),
                    "score": float(s[n - window_size]),
                    "threshold": threshold,
                    "shift_share": shift_share(rows, int(n), window_size),
                }
                for n in alerts
            ]
    return run_rows, alert_rows


def write_rows(path: Path, fields: list[str], profile: str, rows: list[dict]) -> None:
    """Replace this profile's rows in a CSV, keeping other profiles' rows."""
    kept = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            kept = [r for r in csv.DictReader(f) if r["profile"] != profile]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields, restval="")
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)


def summarise(run_rows: list[dict]) -> list[str]:
    """Lines per detector: false alarms on normal_only, then detections and delays per scenario."""
    lines = []
    for detector in DETECTORS:
        mine = [r for r in run_rows if r["detector"] == detector]
        normal = [r["alerts_per_1000"] for r in mine if r["scenario"] == NORMAL]
        lines.append(
            f"{detector}: normal_only {statistics.mean(normal):.2f} alerts per 1,000 requests,"
            f" {sum(r['alerts'] > 0 for r in mine if r['scenario'] == NORMAL)}/{len(normal)}"
            " runs with an alert"
        )
        for name in sorted({r["scenario"] for r in mine} - {NORMAL}):
            delays = [r["delay"] for r in mine if r["scenario"] == name]
            s = delay_summary(delays)
            lines.append(
                f"  {name:<20} detected {s['detected']}/{s['runs']}  median delay {s['median']}"
                f"  range {s['min']}-{s['max']}"
            )
    return lines


def verify(scores: Scores, thresholds: dict[str, float], window_size: int) -> dict:
    """Compare replayed scores with the saved multiseed and embed_baselines results."""
    multiseed = json.loads(MULTISEED_PATH.read_text(encoding="utf-8"))
    embed = json.loads(EMBED_BASELINES_PATH.read_text(encoding="utf-8"))
    saved = {
        SAE: (multiseed["scores"]["sae"], multiseed["detectors"]["sae"]["saved_threshold"]),
        EMBEDDING: (embed["scores"]["mean"], embed["detectors"]["mean"]["saved_threshold"]),
    }
    report = {}
    for detector, (old_scores, old_threshold) in saved.items():
        gaps, same_first = [], 0
        for key, new in scores[detector].items():
            old = np.array(old_scores[f"{key[0]}/{key[1]}"])
            gaps.append(float(np.abs(new - old).max()))
            threshold = thresholds[detector]
            same_first += first_alert(old, threshold, window_size) == first_alert(
                new, threshold, window_size
            )
        report[detector] = {
            "runs": len(gaps),
            "max_abs_diff": max(gaps),
            "same_first_alert": same_first,
            "saved_threshold": old_threshold,
            "threshold": thresholds[detector],
        }
    return report


def main(steps: list[str]) -> None:
    """Run the requested steps for the active profile."""
    profile = active_profile()
    config = load_config()
    runs = run_sequences()
    path = features_path(profile.name)
    if "extract" in steps:
        prompts = needed_prompts(runs)
        print(f"{profile.name}: {len(prompts)} prompts needed")
        extend_cache(EMBEDDINGS_PATH, prompts, encode_embeddings, EMBEDDING_SCHEMA)
        extend_cache(path, prompts, encode_features, FEATURE_SCHEMA)
    if not {"calibrate", "replay", "verify"} & set(steps):
        return
    if not path.exists():
        raise FileNotFoundError(f"no feature cache for {profile.name}; run the extract step")
    hits = load_hits(path)
    if "calibrate" in steps:
        print(f"Calibrated {profile.name}: {calibrate_profile(hits, config)}")
    if not {"replay", "verify"} & set(steps):
        return
    thresholds, counts, mean = detector_thresholds(config.window_size)
    embeddings = load_embeddings(EMBEDDINGS_PATH)
    scores = replay_scores(runs, hits, embeddings, counts, mean, config.window_size)
    if "verify" in steps:
        report = verify(scores, thresholds, config.window_size)
        print(f"Replay vs saved results: {report}")
        for detector, r in report.items():
            if r["same_first_alert"] != r["runs"] or r["max_abs_diff"] > VERIFY_TOLERANCE[detector]:
                raise RuntimeError(f"{detector}: replay does not reproduce the saved results")
    if "replay" in steps:
        run_rows, alert_rows = tabulate(profile.name, runs, scores, thresholds, config.window_size)
        write_rows(RUNS_PATH, RUN_FIELDS, profile.name, run_rows)
        write_rows(ALERTS_PATH, ALERT_FIELDS, profile.name, alert_rows)
        print(
            f"Saved {len(run_rows)} runs to {RUNS_PATH} and {len(alert_rows)} alerts to {ALERTS_PATH}"
        )
        print("\n".join(summarise(run_rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("steps", nargs="+", choices=["extract", "calibrate", "replay", "verify"])
    main(parser.parse_args().steps)
