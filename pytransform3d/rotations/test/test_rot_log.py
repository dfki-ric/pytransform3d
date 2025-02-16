import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_check_skew_symmetric_matrix():
    with pytest.raises(
        ValueError, match="Expected skew-symmetric matrix with shape"
    ):
        pr.check_skew_symmetric_matrix([])
    with pytest.raises(
        ValueError, match="Expected skew-symmetric matrix with shape"
    ):
        pr.check_skew_symmetric_matrix(np.zeros((3, 4)))
    with pytest.raises(
        ValueError, match="Expected skew-symmetric matrix with shape"
    ):
        pr.check_skew_symmetric_matrix(np.zeros((4, 3)))
    V = np.zeros((3, 3))
    V[0, 0] = 0.001
    with pytest.raises(
        ValueError,
        match="Expected skew-symmetric matrix, but it failed the test",
    ):
        pr.check_skew_symmetric_matrix(V)
    with warnings.catch_warnings(record=True) as w:
        pr.check_skew_symmetric_matrix(V, strict_check=False)
        assert len(w) == 1

    pr.check_skew_symmetric_matrix(np.zeros((3, 3)))


def test_cross_product_matrix():
    """Test cross-product matrix."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = pr.random_vector(rng)
        w = pr.random_vector(rng)
        V = pr.cross_product_matrix(v)
        pr.check_skew_symmetric_matrix(V)
        r1 = np.cross(v, w)
        r2 = np.dot(V, w)
        assert_array_almost_equal(r1, r2)
