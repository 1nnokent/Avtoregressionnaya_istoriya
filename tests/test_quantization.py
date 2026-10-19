import numpy as np

from src.quantization import dequantize_uniform, fit_uniform_range, quantize_uniform


def test_tokens_are_int64_in_range():
    x = np.linspace(-3, 3, 1000)
    tokens = quantize_uniform(x, -3.0, 3.0, 256)
    assert tokens.dtype == np.int64
    assert tokens.min() == 0 and tokens.max() == 255


def test_out_of_range_values_are_clipped():
    tokens = quantize_uniform(np.array([-100.0, 100.0]), -1.0, 1.0, 10)
    assert tokens.tolist() == [0, 9]


def test_error_at_most_half_bin():
    x = np.random.default_rng(0).uniform(-2, 5, size=10_000)
    low, high = fit_uniform_range(x)
    restored = dequantize_uniform(quantize_uniform(x, low, high, 64), low, high, 64)
    assert np.abs(restored - x).max() <= (high - low) / 64 / 2 + 1e-12
