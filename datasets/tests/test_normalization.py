import numpy as np
import pytest

from datasets.src.normalization import (
    SCALINGS,
    TRANSFORMS,
    Normalizer,
    inverse_signed_log,
    signed_log,
)


def test_roundtrip():
    x = np.random.default_rng(0).normal(scale=10, size=1000)
    for scale in [1.0, 255.0]:
        np.testing.assert_allclose(
            inverse_signed_log(signed_log(x, scale), scale), x, atol=1e-12
        )


def test_keeps_sign_and_zero():
    y = signed_log(np.array([-5.0, 0.0, 5.0]))
    assert y[0] < 0 and y[1] == 0 and y[2] > 0
    assert y[0] == -y[2]


@pytest.mark.parametrize("transform", TRANSFORMS)
@pytest.mark.parametrize("scaling", SCALINGS)
@pytest.mark.parametrize("per_position", [False, True])
def test_normalizer_roundtrip(transform, scaling, per_position):
    x = np.random.default_rng(0).laplace(scale=[5.0, 0.1, 0.01], size=(500, 3))
    normalizer = Normalizer(
        transform=transform, log_scale=100.0, scaling=scaling, per_position=per_position
    ).fit(x)
    np.testing.assert_allclose(normalizer.inverse(normalizer(x)), x, atol=1e-9)


def test_minmax_maps_train_to_unit_interval():
    x = np.random.default_rng(0).normal(scale=[10.0, 0.1], size=(1000, 2))
    for per_position in [False, True]:
        y = Normalizer(scaling="minmax", per_position=per_position).fit(x)(x)
        assert y.min() == pytest.approx(0) and y.max() == pytest.approx(1)
    y = Normalizer(scaling="minmax", per_position=True).fit(x)(x)
    np.testing.assert_allclose(y.min(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(y.max(axis=0), 1)


def test_unfitted_normalizer_fails():
    with pytest.raises(RuntimeError):
        Normalizer()(np.zeros((2, 2)))
