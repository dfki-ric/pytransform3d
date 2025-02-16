import numpy as np
from numpy.testing import assert_array_almost_equal

import pytransform3d.transformations as pt


def test_jacobian_se3():
    Stheta = np.zeros(6)

    J = pt.left_jacobian_SE3(Stheta)
    J_series = pt.left_jacobian_SE3_series(Stheta, 20)
    assert_array_almost_equal(J, J_series)

    J_inv = pt.left_jacobian_SE3_inv(Stheta)
    J_inv_serias = pt.left_jacobian_SE3_inv_series(Stheta, 20)
    assert_array_almost_equal(J_inv, J_inv_serias)

    J_inv_J = np.dot(J_inv, J)
    assert_array_almost_equal(J_inv_J, np.eye(6))

    rng = np.random.default_rng(0)
    for _ in range(5):
        Stheta = pt.random_exponential_coordinates(rng)

        J = pt.left_jacobian_SE3(Stheta)
        J_series = pt.left_jacobian_SE3_series(Stheta, 20)
        assert_array_almost_equal(J, J_series)

        J_inv = pt.left_jacobian_SE3_inv(Stheta)
        J_inv_serias = pt.left_jacobian_SE3_inv_series(Stheta, 20)
        assert_array_almost_equal(J_inv, J_inv_serias)

        J_inv_J = np.dot(J_inv, J)
        assert_array_almost_equal(J_inv_J, np.eye(6))
