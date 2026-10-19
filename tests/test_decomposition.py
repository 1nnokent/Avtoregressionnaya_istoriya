import numpy as np
import pytest

from src.decomposition import dct2, idct2


@pytest.fixture
def image():
    return np.random.default_rng(0).random((28, 28))


def test_roundtrip(image):
    np.testing.assert_allclose(idct2(dct2(image)), image, atol=1e-12)


def test_energy_preserved(image):
    # orthonormal basis: sum of squares is the same in both domains (Parseval)
    assert np.isclose(np.sum(dct2(image) ** 2), np.sum(image**2))


def test_dc_is_scaled_mean(image):
    # catches a 1D DCT along one axis: round-trip and energy tests would still pass
    assert np.isclose(dct2(image)[0, 0], image.mean() * 28)


def test_constant_image_has_only_dc():
    coeffs = dct2(np.full((28, 28), 0.5))
    coeffs[0, 0] = 0
    np.testing.assert_allclose(coeffs, 0, atol=1e-12)


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        dct2(np.zeros((1, 28, 28)))
