import numpy as np

from src.normalization import inverse_signed_log, signed_log


def test_roundtrip():
    x = np.random.default_rng(0).normal(scale=10, size=1000)
    for scale in [1.0, 255.0]:
        np.testing.assert_allclose(inverse_signed_log(signed_log(x, scale), scale), x, atol=1e-12)


def test_keeps_sign_and_zero():
    y = signed_log(np.array([-5.0, 0.0, 5.0]))
    assert y[0] < 0 and y[1] == 0 and y[2] > 0
    assert y[0] == -y[2]
