import numpy as np
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_jacobian_so3():
    omega = np.zeros(3)

    J = pr.left_jacobian_SO3(omega)
    J_series = pr.left_jacobian_SO3_series(omega, 20)
    assert_array_almost_equal(J, J_series)

    J_inv = pr.left_jacobian_SO3_inv(omega)
    J_inv_series = pr.left_jacobian_SO3_inv_series(omega, 20)
    assert_array_almost_equal(J_inv, J_inv_series)

    J_inv_J = np.dot(J_inv, J)
    assert_array_almost_equal(J_inv_J, np.eye(3))

    rng = np.random.default_rng(0)
    for _ in range(5):
        omega = pr.random_compact_axis_angle(rng)

        J = pr.left_jacobian_SO3(omega)
        J_series = pr.left_jacobian_SO3_series(omega, 20)
        assert_array_almost_equal(J, J_series)

        J_inv = pr.left_jacobian_SO3_inv(omega)
        J_inv_series = pr.left_jacobian_SO3_inv_series(omega, 20)
        assert_array_almost_equal(J_inv, J_inv_series)

        J_inv_J = np.dot(J_inv, J)
        assert_array_almost_equal(J_inv_J, np.eye(3))
