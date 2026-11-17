import numpy as np
import pytest

import datasets.src.kmeans as kmeans_module
from datasets.src.kmeans import assign_codebook, decode_codebook, fit_codebook


def test_fit_codebook_is_deterministic_for_fixed_seed():
    rng = np.random.default_rng(0)
    observations = np.concatenate(
        [
            rng.normal(-2.0, 0.2, size=(100, 3)),
            rng.normal(2.0, 0.2, size=(100, 3)),
        ]
    )

    first = fit_codebook(observations, 4, 120, 10, 17)
    second = fit_codebook(observations, 4, 120, 10, 17)

    np.testing.assert_array_equal(first, second)


def test_fit_codebook_bounds_sample_and_sorts_centroids(monkeypatch):
    captured = {}

    def fake_kmeans2(data, k, **kwargs):
        captured["data"] = data
        captured["k"] = k
        return np.array([[3.0, 0.0], [-1.0, 2.0], [-1.0, 1.0]]), np.zeros(
            len(data), dtype=np.int64
        )

    monkeypatch.setattr(kmeans_module, "kmeans2", fake_kmeans2)
    observations = np.arange(200, dtype=np.float64).reshape(100, 2)

    centroids = fit_codebook(observations, 3, 7, 5, 11)

    assert captured["data"].shape == (7, 2)
    assert captured["k"] == 3
    np.testing.assert_array_equal(
        centroids, np.array([[-1.0, 1.0], [-1.0, 2.0], [3.0, 0.0]])
    )


def test_assign_codebook_uses_nearest_centroid_and_returns_int64():
    observations = np.array([-2.0, 0.0, 1.0, 3.0])
    centroids = np.array([[-1.0], [2.0]])

    tokens = assign_codebook(observations, centroids)

    np.testing.assert_array_equal(tokens, np.array([0, 0, 1, 1]))
    assert tokens.dtype == np.int64


def test_decode_codebook_restores_centroid_vectors():
    centroids = np.array([[-1.0, 2.0, 4.0], [3.0, 5.0, 7.0]])
    tokens = np.array([[1, 0]])

    decoded = decode_codebook(tokens, centroids)

    np.testing.assert_array_equal(
        decoded, np.array([[[3.0, 5.0, 7.0], [-1.0, 2.0, 4.0]]])
    )


@pytest.mark.parametrize("dimensions", [1, 3])
def test_codebook_supports_scalar_and_vector_observations(dimensions):
    rng = np.random.default_rng(3)
    observations = rng.normal(size=(80, dimensions))
    if dimensions == 1:
        observations = observations[:, 0]

    centroids = fit_codebook(observations, 4, 80, 5, 9)
    tokens = assign_codebook(observations, centroids)
    decoded = decode_codebook(tokens, centroids)

    assert centroids.shape == (4, dimensions)
    assert tokens.shape == (80,)
    assert decoded.shape == (80, dimensions)
    assert tokens.min() >= 0 and tokens.max() < 4


def test_codebook_supports_circular_hsv_observations():
    rng = np.random.default_rng(4)
    hue = rng.random(100)
    saturation = rng.random(100)
    value = rng.random(100)
    observations = np.column_stack(
        [
            np.cos(2 * np.pi * hue),
            np.sin(2 * np.pi * hue),
            saturation,
            value,
        ]
    )

    centroids = fit_codebook(observations, 5, 100, 5, 13)
    tokens = assign_codebook(observations, centroids)
    decoded = decode_codebook(tokens, centroids)

    assert centroids.shape == (5, 4)
    assert tokens.shape == (100,)
    assert decoded.shape == (100, 4)


@pytest.mark.parametrize(
    "observations", [np.array([0.0, np.nan]), np.array([[0.0], [np.inf]])]
)
def test_fit_codebook_rejects_non_finite_observations(observations):
    with pytest.raises(ValueError, match="finite"):
        fit_codebook(observations, 1, 2, 5, 0)


def test_assign_codebook_rejects_non_finite_data():
    with pytest.raises(ValueError, match="finite"):
        assign_codebook(np.array([0.0, np.nan]), np.array([[0.0]]))
    with pytest.raises(ValueError, match="finite"):
        assign_codebook(np.array([0.0, 1.0]), np.array([[np.inf]]))


@pytest.mark.parametrize("shape", [(), (2, 2, 2)])
def test_fit_codebook_rejects_invalid_observation_dimensions(shape):
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        fit_codebook(np.zeros(shape), 1, 1, 5, 0)


def test_assign_codebook_rejects_invalid_or_mismatched_dimensions():
    with pytest.raises(ValueError, match="one- or two-dimensional"):
        assign_codebook(np.zeros((2, 2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="two-dimensional"):
        assign_codebook(np.zeros((2, 2)), np.zeros(2))
    with pytest.raises(ValueError, match="feature dimensions"):
        assign_codebook(np.zeros((2, 2)), np.zeros((2, 3)))


@pytest.mark.parametrize("num_observations, sample_size", [(3, 3), (10, 3)])
def test_fit_codebook_rejects_fewer_sampled_observations_than_centroids(
    num_observations, sample_size
):
    with pytest.raises(ValueError, match="sampled observations"):
        fit_codebook(np.arange(num_observations), 4, sample_size, 5, 0)
