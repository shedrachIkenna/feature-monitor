"""Embedding baseline detector: the SAE detector's drift method applied to prompt clusters.

Each prompt is embedded with all-MiniLM-L6-v2 (a vector of 384 numbers that describes its meaning)
and assigned to the nearest of 50 k-means clusters fitted on the baseline prompts. A prompt's
cluster is stored as a one-item ID list, the same shape as an SAE prompt's top-k feature list, so
windows, Jensen-Shannon scoring and threshold calibration reuse monitor.drift and monitor.calibrate
unchanged, on the same prompt pools, window sizes and seeds. The only difference between the two
detectors is the representation.

Run with: uv run python -m monitor.embed_drift
"""

import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from itertools import chain
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from monitor.baseline import (
    BASELINES_DIR,
    POOL_DIR,
    baseline_path,
    load_baseline,
    read_pool,
    sample_prompts,
    save_baseline,
)
from monitor.calibrate import (
    CALIBRATION_POOL,
    CHECK_POOL,
    calibrate_sizes,
    load_pool_features,
    load_threshold,
    pool_prompts,
    print_sweep,
    save_threshold,
    threshold_path,
)
from monitor.config import load_config
from monitor.drift import score_window

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_CLUSTERS = 50
KMEANS_SEED = 0
NAME = f"minilm_k{N_CLUSTERS}"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results" / "embed_jsd.json"


@lru_cache(maxsize=1)
def load_embedder() -> SentenceTransformer:
    """Load the sentence embedding model once, on CPU."""
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def embed_prompts(prompts: Sequence[str]) -> np.ndarray:
    """Return one unit-length embedding per prompt, shape (len(prompts), 384)."""
    return load_embedder().encode(
        list(prompts),
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(prompts) > 1000,
    )


def embeddings_cache_path(pool: str) -> Path:
    """Return the (uncommitted) embedding cache for one pool."""
    return BASELINES_DIR / f"minilm_{pool}_embeddings.npy"


def load_or_embed(prompts: Sequence[str], cache: Path) -> np.ndarray:
    """Return the prompts' embeddings, reusing the cache if it has one row per prompt."""
    if cache.exists():
        embeddings = np.load(cache)
        if len(embeddings) == len(prompts):
            return embeddings
    embeddings = embed_prompts(prompts)
    np.save(cache, embeddings)
    return embeddings


def fit_centroids(
    embeddings: np.ndarray, k: int = N_CLUSTERS, seed: int = KMEANS_SEED
) -> np.ndarray:
    """Fit k-means with a fixed seed and return the cluster centres, shape (k, dim)."""
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embeddings).cluster_centers_


def assign_clusters(embeddings: np.ndarray, centroids: np.ndarray) -> list[list[int]]:
    """Put each embedding in its nearest cluster, as a one-item list like an SAE feature list."""
    # Squared distance |x - c|^2 = |x|^2 - 2 x.c + |c|^2; |x|^2 is the same for every c.
    distances = (centroids**2).sum(axis=1) - 2 * embeddings @ centroids.T
    return [[int(c)] for c in distances.argmin(axis=1)]


def cluster_counts(cluster_sets: Sequence[Sequence[int]]) -> dict[int, int]:
    """Count prompts per cluster, sorted by cluster ID."""
    return dict(sorted(Counter(chain.from_iterable(cluster_sets)).items()))


def main() -> None:
    """Fit clusters on the SAE baseline's prompts, calibrate, and score one window with both."""
    config = load_config()
    sae_path = baseline_path()
    sae_metadata, sae_baseline = load_baseline(sae_path)

    train = read_pool(POOL_DIR / f"{sae_metadata['pool']}.jsonl")
    baseline_prompts = sample_prompts(train, sae_metadata["prompt_count"], sae_metadata["seed"])
    baseline_embeddings = load_or_embed(baseline_prompts, embeddings_cache_path("baseline"))
    centroids = fit_centroids(baseline_embeddings)
    counts = cluster_counts(assign_clusters(baseline_embeddings, centroids))

    pools = {
        name: assign_clusters(load_or_embed(prompts, embeddings_cache_path(name)), centroids)
        for name, prompts in pool_prompts(sae_metadata).items()
    }
    results = calibrate_sizes(pools[CALIBRATION_POOL], pools[CHECK_POOL], counts, config)

    metadata = {
        "embed_model": EMBED_MODEL,
        "n_clusters": N_CLUSTERS,
        "kmeans_seed": KMEANS_SEED,
        "pool": sae_metadata["pool"],
        "prompt_count": len(baseline_prompts),
        "seed": sae_metadata["seed"],
        "sae_baseline_created": sae_metadata["created"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = BASELINES_DIR / f"{NAME}.json"
    rates = {c: n / len(baseline_prompts) * 1000 for c, n in counts.items()}
    save_baseline(
        {"prompt_count": len(baseline_prompts), "counts": counts, "rate_per_1000": rates},
        metadata,
        path,
    )
    np.save(path.with_name(f"{NAME}_centroids.npy"), centroids)
    pool_sizes = {name: len(sets) for name, sets in pools.items()}
    save_threshold(results[config.window_size], config, metadata, pool_sizes, threshold_path(path))
    embed_threshold = load_threshold(threshold_path(path), config.window_size, metadata["created"])

    RESULTS_PATH.write_text(
        json.dumps(
            {"config": vars(config), "pool_sizes": pool_sizes, "results": list(results.values())}
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {path}, its centroids and threshold, and {RESULTS_PATH}")
    print(f"Pools: {pool_sizes}; baseline clusters used: {len(counts)} of {N_CLUSTERS}")
    print_sweep(results, config.window_size)

    # The same window for both detectors: the first window_size prompts of the check pool.
    sae_sets = load_pool_features(sae_path, sae_metadata)[CHECK_POOL][: config.window_size]
    sae_threshold = load_threshold(
        threshold_path(sae_path), config.window_size, sae_metadata["created"]
    )
    window = {
        "SAE features": (score_window(sae_baseline["counts"], sae_sets), sae_threshold),
        "embedding clusters": (
            score_window(counts, pools[CHECK_POOL][: config.window_size]),
            embed_threshold,
        ),
    }
    print(f"\nFirst {config.window_size} {CHECK_POOL} prompts:")
    for detector, (score, threshold) in window.items():
        print(f"  {detector:<18}  JSD {score:.4f}  threshold {threshold:.4f}")


if __name__ == "__main__":
    main()
