import numpy as np
import pytest

from datasets.src.quantization import KMeansQuantizer, Quantizer


def test_tokens_are_int64_in_range():
    x = np.linspace(-3, 3, 1000)
    tokens = Quantizer(num_bins=256).fit(x).encode(x)
    assert tokens.dtype == np.int64
    assert tokens.min() == 0 and tokens.max() == 255


def test_out_of_range_values_are_clipped():
    quantizer = Quantizer(num_bins=10).fit(np.array([-1.0, 1.0]))
    tokens = quantizer.encode(np.array([-100.0, 100.0]))
    assert tokens.tolist() == [0, 9]


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        Quantizer(method="quantile")


def test_unfitted_quantizer_fails():
    with pytest.raises(RuntimeError):
        Quantizer().encode(np.zeros(2))


def test_error_at_most_half_bin():
    x = np.random.default_rng(0).uniform(-2, 5, size=10_000)
    quantizer = Quantizer(num_bins=64).fit(x)
    restored = quantizer.decode(quantizer.encode(x))
    assert np.abs(restored - x).max() <= (x.max() - x.min()) / 64 / 2 + 1e-12


@pytest.mark.parametrize("per_position", [False, True])
def test_quantizer_contract_and_error_decreases(per_position):
    x = np.random.default_rng(0).laplace(scale=[1.0, 0.1, 3.0], size=(2000, 3))
    errors = []
    for num_bins in [8, 32, 128]:
        quantizer = Quantizer(num_bins=num_bins, per_position=per_position).fit(x)
        tokens = quantizer.encode(x)
        assert tokens.dtype == np.int64 and tokens.shape == x.shape
        assert tokens.min() >= 0 and tokens.max() < num_bins
        errors.append(np.mean((quantizer.decode(tokens) - x) ** 2))
    assert errors[0] > errors[1] > errors[2]


def test_per_position_shapes():
    quantizer = Quantizer(num_bins=10, per_position=True).fit(
        np.random.default_rng(0).random((100, 7))
    )
    assert quantizer.thresholds.shape == (7, 9) and quantizer.centers.shape == (7, 10)


def test_state_roundtrip_preserves_tokens_and_decoding():
    train = np.array([[-2.0, 0.0], [1.0, 4.0]])
    values = np.array([[-1.0, 2.0]])
    fitted = Quantizer(num_bins=4, per_position=True).fit(train)

    restored = Quantizer(num_bins=4, per_position=True)
    restored.load_state_dict(fitted.state_dict())

    expected_tokens = fitted.encode(values)
    np.testing.assert_array_equal(restored.encode(values), expected_tokens)
    np.testing.assert_array_equal(
        restored.decode(expected_tokens), fitted.decode(expected_tokens)
    )


@pytest.fixture
def scalar_coefficients():
    rng = np.random.default_rng(5)
    return np.stack(
        [
            rng.normal(-4.0, 0.5, size=(80, 6)),
            rng.normal(0.0, 0.5, size=(80, 6)),
            rng.normal(4.0, 0.5, size=(80, 6)),
        ],
        axis=1,
    )


def test_kmeans_scalar_fits_one_codebook_per_channel(scalar_coefficients):
    quantizer = KMeansQuantizer(
        num_bins=4, sample_size=300, iterations=10, seed=3
    ).fit(scalar_coefficients)

    assert quantizer.centers.shape == (3, 4, 1)
    assert quantizer.channels == 3
    assert quantizer.positions == 6
    assert quantizer.centers[0].mean() < quantizer.centers[1].mean()
    assert quantizer.centers[1].mean() < quantizer.centers[2].mean()


def test_kmeans_scalar_uses_channel_major_token_order_and_decodes_shape():
    quantizer = KMeansQuantizer(num_bins=2)
    quantizer.load_state_dict(
        {
            "centers": np.array([[[0.0], [10.0]], [[100.0], [110.0]]]),
            "channels": np.asarray(2, dtype=np.int64),
            "positions": np.asarray(2, dtype=np.int64),
        }
    )
    coefficients = np.array([[[0.0, 10.0], [110.0, 100.0]]])

    tokens = quantizer.encode(coefficients)
    decoded = quantizer.decode(tokens)

    np.testing.assert_array_equal(tokens, np.array([[0, 1, 1, 0]]))
    np.testing.assert_array_equal(decoded, coefficients)
    assert decoded.shape == (1, 2, 2)


def test_kmeans_scalar_fit_is_deterministic(scalar_coefficients):
    params = {"num_bins": 6, "sample_size": 250, "iterations": 10, "seed": 7}

    first = KMeansQuantizer(**params).fit(scalar_coefficients)
    second = KMeansQuantizer(**params).fit(scalar_coefficients)

    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(
        first.encode(scalar_coefficients), second.encode(scalar_coefficients)
    )


def test_kmeans_scalar_distortion_decreases_with_more_centroids():
    coefficients = np.random.default_rng(8).normal(size=(200, 3, 8))
    distortions = []

    for num_bins in [2, 4, 8]:
        quantizer = KMeansQuantizer(
            num_bins=num_bins,
            sample_size=2000,
            iterations=20,
            seed=9,
        ).fit(coefficients)
        restored = quantizer.decode(quantizer.encode(coefficients))
        distortions.append(np.mean((restored - coefficients) ** 2))

    assert distortions[0] > distortions[1] > distortions[2]


def test_kmeans_scalar_rejects_channel_mismatch(scalar_coefficients):
    quantizer = KMeansQuantizer(
        num_bins=4, sample_size=300, iterations=5, seed=1
    ).fit(scalar_coefficients)

    with pytest.raises(ValueError, match="3 channels"):
        quantizer.encode(scalar_coefficients[:, :2])


def test_kmeans_scalar_state_roundtrip_preserves_results(scalar_coefficients):
    fitted = KMeansQuantizer(
        num_bins=4, sample_size=300, iterations=10, seed=2
    ).fit(scalar_coefficients)
    restored = KMeansQuantizer(num_bins=4)

    restored.load_state_dict(fitted.state_dict())

    expected_tokens = fitted.encode(scalar_coefficients)
    np.testing.assert_array_equal(restored.encode(scalar_coefficients), expected_tokens)
    np.testing.assert_array_equal(
        restored.decode(expected_tokens), fitted.decode(expected_tokens)
    )


def test_kmeans_scalar_rejects_non_ncl_coefficients():
    with pytest.raises(ValueError, match="N, C, L"):
        KMeansQuantizer(num_bins=2).fit(np.zeros((10, 4)))


def test_kmeans_scalar_rejects_unknown_grouping():
    with pytest.raises(ValueError, match="unknown KMeans grouping"):
        KMeansQuantizer(grouping="spatial")
