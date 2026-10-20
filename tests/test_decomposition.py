import numpy as np
import pytest

from src.decomposition import DECOMPOSITIONS, get_decomposition


@pytest.fixture
def image():
    return np.random.default_rng(0).random((28, 28))


def test_roundtrip(image):
    decomposition = get_decomposition("dct")
    np.testing.assert_allclose(
        decomposition.inverse(decomposition.forward(image)), image, atol=1e-12
    )


def test_energy_preserved(image):
    # orthonormal basis: sum of squares is the same in both domains (Parseval)
    coeffs = get_decomposition("dct").forward(image)
    assert np.isclose(np.sum(coeffs**2), np.sum(image**2))


def test_dc_is_scaled_mean(image):
    # catches a 1D DCT along one axis: round-trip and energy tests would still pass
    assert np.isclose(get_decomposition("dct").forward(image)[0, 0], image.mean() * 28)


def test_constant_image_has_only_dc():
    coeffs = get_decomposition("dct").forward(np.full((28, 28), 0.5))
    coeffs[0, 0] = 0
    np.testing.assert_allclose(coeffs, 0, atol=1e-12)


@pytest.mark.parametrize("name", DECOMPOSITIONS)
@pytest.mark.parametrize("shape", [(5, 28, 28), (2, 3, 32, 32), (3, 16, 8)])
def test_batched_roundtrip_and_energy(name, shape):
    x = np.random.default_rng(0).random(shape)
    decomposition = (
        get_decomposition(name, block_size=4)
        if name == "block_dct"
        else get_decomposition(name)
    )
    coeffs = decomposition.forward(x)
    assert coeffs.shape == x.shape
    np.testing.assert_allclose(decomposition.inverse(coeffs), x, atol=1e-10)
    np.testing.assert_allclose((coeffs**2).sum(), (x**2).sum(), rtol=1e-10)


@pytest.mark.parametrize("name", ["dct", "dft"])
def test_constant_image_has_only_lowest_frequency(name):
    coeffs = get_decomposition(name).forward(np.full((16, 16), 0.5))
    assert np.isclose(coeffs[0, 0], 0.5 * 16)
    coeffs[0, 0] = 0
    assert np.allclose(coeffs, 0)


def test_dft_low_frequencies_at_top_left():
    # one horizontal period: frequencies +1 and -1 must land in the first columns, not at the far end
    x = np.cos(2 * np.pi * np.arange(16) / 16)[None, :].repeat(16, axis=0)
    energy = get_decomposition("dft").forward(x) ** 2
    assert energy[0, 1:3].sum() == pytest.approx(energy.sum())


def test_wavelet_params():
    x = np.random.default_rng(0).random((2, 32, 32))
    decomposition = get_decomposition("wavelet", wavelet="db2", level=2)
    np.testing.assert_allclose(
        decomposition.inverse(decomposition.forward(x)), x, atol=1e-10
    )


def test_unknown_decomposition():
    with pytest.raises(ValueError):
        get_decomposition("fourier")


def test_block_dct_with_one_block_is_global_dct():
    x = np.random.default_rng(0).random((2, 16, 16))
    np.testing.assert_allclose(
        get_decomposition("block_dct", block_size=16).forward(x),
        get_decomposition("dct").forward(x),
    )


def test_block_dct_transforms_blocks_independently():
    x = np.random.default_rng(0).random((16, 16))
    coeffs = get_decomposition("block_dct", block_size=8).forward(x)
    expected = get_decomposition("dct").forward(x[8:, :8])
    np.testing.assert_allclose(coeffs[8:, :8], expected)


def test_block_dct_rejects_indivisible_size():
    with pytest.raises(ValueError):
        get_decomposition("block_dct", block_size=8).forward(np.zeros((28, 28)))
