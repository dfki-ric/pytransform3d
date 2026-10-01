import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.linalg import expm

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


@pytest.mark.parametrize("translation_scale", [1e-12, 1.0, 1e6])
def test_jacobian_se3_pure_translation(translation_scale):
    translation = translation_scale * np.array([2.0, -1.0, 3.0])
    Stheta = np.r_[np.zeros(3), translation]
    translation_cross = np.cross(translation, np.eye(3), axis=0)
    expected = np.eye(6)
    expected[3:, :3] = 0.5 * translation_cross
    expected_inv = np.eye(6)
    expected_inv[3:, :3] = -0.5 * translation_cross

    # The adjoint algebra matrix squares to zero for pure translations.
    with np.errstate(divide="raise", invalid="raise"):
        J = pt.left_jacobian_SE3(Stheta)
        J_inv = pt.left_jacobian_SE3_inv(Stheta)

    assert_allclose(J, expected, rtol=0.0, atol=0.0)
    assert_allclose(J_inv, expected_inv, rtol=0.0, atol=0.0)
    assert_allclose(J_inv.dot(J), np.eye(6), rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "angle",
    [
        1e-18,
        1e-9,
        4e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        0.1,
        0.49,
        0.5,
        0.51,
        1.0,
        np.pi,
    ],
)
@pytest.mark.parametrize("translation_scale", [1e-12, 1.0, 1e6])
def test_jacobian_se3_against_matrix_exponential(angle, translation_scale):
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    rotation = angle * axis
    translation = np.array([2.0, -1.0, 3.0])

    # exp([[ad(x), I], [0, 0]]) has integral_0^1 exp(t*ad(x)) dt
    # in its upper-right block. This evaluates the Jacobian independently
    # of both the closed form and the library's Taylor-series functions.
    rotation_cross = np.cross(rotation, np.eye(3), axis=0)
    translation_cross = np.cross(translation, np.eye(3), axis=0)
    generator = np.zeros((12, 12))
    generator[:3, :3] = rotation_cross
    generator[3:6, :3] = translation_cross
    generator[3:6, 3:6] = rotation_cross
    generator[:6, 6:] = np.eye(6)
    expected = expm(generator)[:6, 6:]
    expected_inv = np.linalg.inv(expected)

    Stheta = np.r_[rotation, translation_scale * translation]
    J = pt.left_jacobian_SE3(Stheta)
    J_inv = pt.left_jacobian_SE3_inv(Stheta)
    # The coupling blocks are linear in translation. Compare them at unit
    # scale so the same precision is required for small and large translations.
    J[3:, :3] /= translation_scale
    J_inv[3:, :3] /= translation_scale
    assert_allclose(J, expected, rtol=1e-14, atol=2e-14)
    assert_allclose(J_inv, expected_inv, rtol=1e-14, atol=2e-14)
    assert_allclose(J_inv.dot(J), np.eye(6), rtol=0.0, atol=2e-14)
