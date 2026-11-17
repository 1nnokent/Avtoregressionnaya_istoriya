import numpy as np
from scipy.cluster.vq import kmeans2, vq


def fit_codebook(
    observations: np.ndarray,
    num_centroids: int,
    sample_size: int,
    iterations: int,
    seed: int,
) -> np.ndarray:
    observations = _as_observations(observations)
    sample_count = min(len(observations), sample_size)
    if sample_count < num_centroids:
        raise ValueError(
            f"need at least {num_centroids} sampled observations, got {sample_count}"
        )

    rng = np.random.default_rng(seed)
    if sample_count < len(observations):
        indices = rng.choice(len(observations), size=sample_count, replace=False)
        sample = observations[indices]
    else:
        sample = observations

    centroids, _ = kmeans2(
        sample,
        num_centroids,
        iter=iterations,
        minit="++",
        missing="raise",
        check_finite=False,
        rng=rng,
    )
    order = np.lexsort(
        tuple(centroids[:, index] for index in range(centroids.shape[1] - 1, -1, -1))
    )
    return np.asarray(centroids[order], dtype=np.float64)


def assign_codebook(observations: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    observations = _as_observations(observations)
    centroids = _as_centroids(centroids)
    if observations.shape[1] != centroids.shape[1]:
        raise ValueError(
            "observations and centroids must have matching feature dimensions"
        )
    tokens, _ = vq(observations, centroids, check_finite=False)
    return np.asarray(tokens, dtype=np.int64)


def decode_codebook(tokens: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    centroids = _as_centroids(centroids)
    return centroids[np.asarray(tokens)]


def _as_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim == 1:
        observations = observations[:, None]
    elif observations.ndim != 2:
        raise ValueError(
            f"observations must be one- or two-dimensional, got shape {observations.shape}"
        )
    if not np.isfinite(observations).all():
        raise ValueError("observations must contain only finite values")
    return observations


def _as_centroids(centroids: np.ndarray) -> np.ndarray:
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.ndim != 2:
        raise ValueError(f"centroids must be two-dimensional, got shape {centroids.shape}")
    if not np.isfinite(centroids).all():
        raise ValueError("centroids must contain only finite values")
    return centroids
