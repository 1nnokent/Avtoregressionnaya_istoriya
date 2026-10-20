import numpy as np
import pytest

from src.quantization import Quantizer


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
