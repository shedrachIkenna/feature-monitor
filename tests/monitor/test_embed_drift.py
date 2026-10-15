import numpy as np
import pytest

from monitor.calibrate import calibrate
from monitor.drift import score_window
from monitor.embed_drift import assign_clusters, cluster_counts, embed_prompts, fit_centroids

N_BLOBS = 5
DIM = 8


def blobs(n: int, seed: int, weights: list[float] | None = None) -> np.ndarray:
    """Points around N_BLOBS well-separated centres, picked with the given blob weights."""
    rng = np.random.default_rng(seed)
    centres = np.eye(N_BLOBS, DIM) * 10
    which = rng.choice(N_BLOBS, size=n, p=weights)
    return centres[which] + rng.normal(scale=0.5, size=(n, DIM))


def test_assign_clusters_picks_the_nearest_centroid() -> None:
    centroids = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    points = np.array([[1.0, 1.0], [9.0, -1.0], [-1.0, 8.0], [6.0, 0.0]])

    assert assign_clusters(points, centroids) == [[0], [1], [2], [1]]


def test_fit_centroids_is_reproducible_and_finds_the_blobs() -> None:
    points = blobs(1000, seed=0)

    first = fit_centroids(points, k=N_BLOBS, seed=0)
    again = fit_centroids(points, k=N_BLOBS, seed=0)

    # Parallel summation inside k-means can differ in the last bit between runs.
    np.testing.assert_allclose(first, again, rtol=0, atol=1e-9)
    assert assign_clusters(points, first) == assign_clusters(points, again)
    assert len(cluster_counts(assign_clusters(points, first))) == N_BLOBS


def test_shifted_window_scores_higher_than_a_normal_one() -> None:
    baseline = blobs(3000, seed=0)
    centroids = fit_centroids(baseline, k=N_BLOBS, seed=0)
    counts = cluster_counts(assign_clusters(baseline, centroids))
    normal = assign_clusters(blobs(500, seed=1), centroids)
    shifted = assign_clusters(blobs(500, seed=2, weights=[0.6, 0.1, 0.1, 0.1, 0.1]), centroids)

    assert score_window(counts, shifted) > 10 * score_window(counts, normal)


def test_cluster_lists_go_through_the_shared_calibration() -> None:
    centroids = fit_centroids(blobs(3000, seed=0), k=N_BLOBS, seed=0)
    counts = cluster_counts(assign_clusters(blobs(3000, seed=0), centroids))
    calibration = assign_clusters(blobs(4000, seed=1), centroids)
    check = assign_clusters(blobs(4000, seed=2), centroids)

    result = calibrate(calibration, check, counts, 100, 99, 500, seed=0)

    assert result["threshold"] > 0
    assert result["check_exceed_rate"] < 0.05


@pytest.mark.requires_model
def test_embed_prompts_are_unit_length_and_meaningful() -> None:
    cat, kitten, stocks = embed_prompts(
        ["The cat sat on the mat.", "A kitten rested on the rug.", "Stock markets fell sharply."]
    )

    assert cat.shape == (384,)
    np.testing.assert_allclose(np.linalg.norm([cat, kitten, stocks], axis=1), 1, atol=1e-5)
    assert cat @ kitten > cat @ stocks
