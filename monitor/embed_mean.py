"""Mean-embedding drift: how far a window's average prompt embedding is from the baseline's.

The live embedding detector scores each window with the squared distance between the mean
all-MiniLM-L6-v2 embedding of its prompts and the mean of the 3,000 baseline prompts. In the
offline evaluation (eval/embed_baselines.py) this detected every shift sooner than 50 clusters.
The 50 clusters (monitor/embed_drift.py) are still used to say which topics changed in an alert.

The threshold is calibrated like the others: the configured percentile of 1,000 random windows
from the normal_train prompts outside the baseline, with the same generator and draw order as
monitor.calibrate, then checked on windows from normal_test.

Run with: uv run python -m monitor.embed_mean   (after monitor.embed_drift)
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from monitor.baseline import (
    BASELINES_DIR,
    POOL_DIR,
    baseline_path,
    load_baseline,
    read_pool,
    sample_prompts,
)
from monitor.calibrate import (
    CALIBRATION_POOL,
    CHECK_POOL,
    load_threshold,
    pool_prompts,
    save_threshold,
    threshold_path,
)
from monitor.config import load_config
from monitor.embed_drift import EMBED_MODEL, embeddings_cache_path, load_or_embed

NAME = "minilm_mean"


def mean_distance(window: np.ndarray, baseline_mean: np.ndarray) -> float:
    """Squared distance between the window's mean embedding and the baseline's."""
    return float(((window.mean(axis=0) - baseline_mean) ** 2).sum())


def mean_baseline_path() -> Path:
    """Return monitor/baselines/minilm_mean.json; its threshold file sits next to it."""
    return BASELINES_DIR / f"{NAME}.json"


def load_mean_baseline(path: Path) -> tuple[dict, np.ndarray]:
    """Read (metadata, baseline mean embedding)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["metadata"], np.array(payload["mean"], dtype=float)


def calibrate_mean(
    calibration: np.ndarray,
    check: np.ndarray,
    baseline_mean: np.ndarray,
    window_size: int,
    percentile: float,
    n_windows: int,
    seed: int,
) -> dict:
    """Threshold from random calibration windows; exceed rate on random check windows."""
    rng = np.random.default_rng([seed, window_size])

    def scores(pool: np.ndarray) -> np.ndarray:
        indices = np.arange(len(pool))
        return np.array(
            [
                mean_distance(
                    pool[rng.choice(indices, size=window_size, replace=False)], baseline_mean
                )
                for _ in range(n_windows)
            ]
        )

    calibration_scores = scores(calibration)
    check_scores = scores(check)
    threshold = float(np.percentile(calibration_scores, percentile))
    return {
        "window_size": window_size,
        "threshold": threshold,
        "check_exceed_rate": float((check_scores > threshold).mean()),
    }


def main() -> None:
    """Save the baseline mean embedding and its calibrated threshold."""
    config = load_config()
    sae_metadata, _ = load_baseline(baseline_path())
    train = read_pool(POOL_DIR / f"{sae_metadata['pool']}.jsonl")
    prompts = sample_prompts(train, sae_metadata["prompt_count"], sae_metadata["seed"])
    baseline = load_or_embed(prompts, embeddings_cache_path("baseline"))
    pools = {
        name: load_or_embed(prompts, embeddings_cache_path(name)).astype(float)
        for name, prompts in pool_prompts(sae_metadata).items()
    }
    baseline_mean = baseline.astype(float).mean(axis=0)
    result = calibrate_mean(
        pools[CALIBRATION_POOL],
        pools[CHECK_POOL],
        baseline_mean,
        config.window_size,
        config.threshold_percentile,
        config.calibration_windows,
        config.seed,
    )

    metadata = {
        "embed_model": EMBED_MODEL,
        "pool": sae_metadata["pool"],
        "prompt_count": len(baseline),
        "seed": sae_metadata["seed"],
        "sae_baseline_created": sae_metadata["created"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = mean_baseline_path()
    path.write_text(
        json.dumps({"metadata": metadata, "mean": baseline_mean.tolist()}) + "\n", encoding="utf-8"
    )
    pool_sizes = {name: len(pool) for name, pool in pools.items()}
    save_threshold(result, config, metadata, pool_sizes, threshold_path(path))
    saved = load_threshold(threshold_path(path), config.window_size, metadata["created"])
    print(f"Saved {path} and its threshold")
    print(
        f"threshold {saved:.6g}; {CHECK_POOL} windows above it: {result['check_exceed_rate']:.1%}"
    )


if __name__ == "__main__":
    main()
