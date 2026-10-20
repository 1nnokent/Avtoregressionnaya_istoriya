import numpy as np
import pytest
from scipy.linalg import sqrtm

from src.metrics.scores import compute_fid, compute_is, fraction_inside, knn_radii


@pytest.mark.parametrize("count,dimension", [(32, 4), (4, 32)])
def test_frechet_matches_covariance_formula(count, dimension):
    rng = np.random.default_rng(4)
    real, fake = rng.normal(size=(count, dimension)), rng.normal(size=(count + 1, dimension))
    ca, cb = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    expected = np.sum((real.mean(0) - fake.mean(0)) ** 2) + np.trace(ca + cb - 2 * sqrtm(ca @ cb).real)
    assert compute_fid(real, fake) == pytest.approx(expected, abs=1e-5)
    assert compute_fid(real, real) == pytest.approx(0, abs=1e-5)


def test_classifier_score_limits():
    assert compute_is(np.full((10, 10), 0.1)) == pytest.approx(1)
    assert compute_is(np.eye(10)) == pytest.approx(10)


def test_knn_geometry_and_rejections():
    features = np.array([[0, 0], [1, 0], [3, 0], [6, 0]])
    radii = knn_radii(features, 1, chunk_size=2)
    np.testing.assert_allclose(radii, [1, 1, 2, 3])
    assert fraction_inside(features, features, radii) == 1
    assert fraction_inside(features + 100, features, radii) == 0
    with pytest.raises(ValueError):
        knn_radii(features, 4)
