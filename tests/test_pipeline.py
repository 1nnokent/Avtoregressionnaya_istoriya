import numpy as np
import pytest

from src.pipeline import fit, image_to_sequence, sequence_to_image, image_to_tokens, tokens_to_image


@pytest.fixture
def images():
    # MNIST-like: mostly black background, a bright blob in the middle
    rng = np.random.default_rng(0)
    images = np.zeros((50, 28, 28))
    images[:, 8:20, 10:18] = rng.random((50, 12, 8))
    return images


def test_sequence_roundtrip_without_quantization(images):
    params = fit(images)
    image = images[0]
    np.testing.assert_allclose(sequence_to_image(image_to_sequence(image, params.log_scale), params), image, atol=1e-10)


def test_tokens_contract(images):
    params = fit(images, num_bins=256)
    tokens = image_to_tokens(images[0], params)
    assert tokens.shape == (28 * 28,)
    assert tokens.dtype == np.int64
    assert 0 <= tokens.min() and tokens.max() < 256


def test_reconstruction_improves_with_more_bins(images):
    errors = []
    for num_bins in [16, 64, 256]:
        params = fit(images, num_bins=num_bins)
        restored = tokens_to_image(image_to_tokens(images[0], params), params)
        errors.append(np.mean((restored - images[0]) ** 2))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-3
