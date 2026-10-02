import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.linalg import expm

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


@pytest.mark.parametrize(
    "angle", [1e-18, 1e-9, 4e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 0.5, np.pi]
)
@pytest.mark.parametrize("axis", [[1.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
def test_jacobian_so3_against_matrix_exponential(angle, axis):
    omega = angle * np.array(axis) / np.linalg.norm(axis)
    generator = np.zeros((6, 6))
    generator[:3, :3] = np.cross(omega, np.eye(3), axis=0)
    generator[:3, 3:] = np.eye(3)
    expected = expm(generator)[:3, 3:]

    J = pr.left_jacobian_SO3(omega)
    J_inv = pr.left_jacobian_SO3_inv(omega)

    assert_allclose(J, expected, rtol=1e-14, atol=2e-15)
    assert_allclose(J_inv, np.linalg.inv(expected), rtol=1e-14, atol=2e-15)
    assert_allclose(J_inv.dot(J), np.eye(3), rtol=0.0, atol=2e-15)
