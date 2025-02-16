import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_active_matrix_from_intrinsic_euler_zxz():
    """Test conversion from intrinsic zxz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz([0.5 * np.pi, 0, 0]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz(
            [0.5 * np.pi, 0, 0.5 * np.pi]
        ),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0]
        ),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        ),
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    )


def test_active_matrix_from_extrinsic_euler_zxz():
    """Test conversion from extrinsic zxz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz([0.5 * np.pi, 0, 0]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz(
            [0.5 * np.pi, 0, 0.5 * np.pi]
        ),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0]
        ),
        np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        ),
        np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    )


def test_active_matrix_from_intrinsic_euler_zyz():
    """Test conversion from intrinsic zyz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        pr.matrix_from_euler([0.5 * np.pi, 0, 0], 2, 1, 2, False),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz(
            [0.5 * np.pi, 0, 0.5 * np.pi]
        ),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0]
        ),
        np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        ),
        np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    )


def test_active_matrix_from_extrinsic_euler_zyz():
    """Test conversion from roll, pitch, yaw."""
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw([0.5 * np.pi, 0, 0]),
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw([0.5 * np.pi, 0, 0]),
        pr.matrix_from_euler([0.5 * np.pi, 0, 0], 0, 1, 2, True),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw(
            [0.5 * np.pi, 0, 0.5 * np.pi]
        ),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw(
            [0.5 * np.pi, 0.5 * np.pi, 0]
        ),
        np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        ),
        np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    )


def test_active_matrix_from_intrinsic_zyx():
    """Test conversion from intrinsic zyx Euler angles."""
    rng = np.random.default_rng(844)
    for _ in range(5):
        euler_zyx = (rng.random(3) - 0.5) * np.array(
            [np.pi, 0.5 * np.pi, np.pi]
        )
        s = np.sin(euler_zyx)
        c = np.cos(euler_zyx)
        R_from_formula = np.array(
            [
                [
                    c[0] * c[1],
                    c[0] * s[1] * s[2] - s[0] * c[2],
                    c[0] * s[1] * c[2] + s[0] * s[2],
                ],
                [
                    s[0] * c[1],
                    s[0] * s[1] * s[2] + c[0] * c[2],
                    s[0] * s[1] * c[2] - c[0] * s[2],
                ],
                [-s[1], c[1] * s[2], c[1] * c[2]],
            ]
        )  # See Lynch, Park: Modern Robotics, page 576

        # Normal case, we can reconstruct original angles
        R = pr.active_matrix_from_intrinsic_euler_zyx(euler_zyx)
        assert_array_almost_equal(R_from_formula, R)
        euler_zyx2 = pr.intrinsic_euler_zyx_from_active_matrix(R)
        assert_array_almost_equal(euler_zyx, euler_zyx2)

        # Gimbal lock 1, infinite solutions with constraint
        # alpha - gamma = constant
        euler_zyx[1] = 0.5 * np.pi
        R = pr.active_matrix_from_intrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.intrinsic_euler_zyx_from_active_matrix(R)
        assert pytest.approx(euler_zyx2[1]) == 0.5 * np.pi
        assert (
            pytest.approx(euler_zyx[0] - euler_zyx[2])
            == euler_zyx2[0] - euler_zyx2[2]
        )

        # Gimbal lock 2, infinite solutions with constraint
        # alpha + gamma = constant
        euler_zyx[1] = -0.5 * np.pi
        R = pr.active_matrix_from_intrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.intrinsic_euler_zyx_from_active_matrix(R)
        assert pytest.approx(euler_zyx2[1]) == -0.5 * np.pi
        assert (
            pytest.approx(euler_zyx[0] + euler_zyx[2])
            == euler_zyx2[0] + euler_zyx2[2]
        )


def test_active_matrix_from_extrinsic_zyx():
    """Test conversion from extrinsic zyx Euler angles."""
    rng = np.random.default_rng(844)
    for _ in range(5):
        euler_zyx = (rng.random(3) - 0.5) * np.array(
            [np.pi, 0.5 * np.pi, np.pi]
        )

        # Normal case, we can reconstruct original angles
        R = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.extrinsic_euler_zyx_from_active_matrix(R)
        assert_array_almost_equal(euler_zyx, euler_zyx2)
        R2 = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx2)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 1, infinite solutions with constraint
        # alpha + gamma = constant
        euler_zyx[1] = 0.5 * np.pi
        R = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.extrinsic_euler_zyx_from_active_matrix(R)
        assert pytest.approx(euler_zyx2[1]) == 0.5 * np.pi
        assert (
            pytest.approx(euler_zyx[0] + euler_zyx[2])
            == euler_zyx2[0] + euler_zyx2[2]
        )
        R2 = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx2)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 2, infinite solutions with constraint
        # alpha - gamma = constant
        euler_zyx[1] = -0.5 * np.pi
        R = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.extrinsic_euler_zyx_from_active_matrix(R)
        assert pytest.approx(euler_zyx2[1]) == -0.5 * np.pi
        assert (
            pytest.approx(euler_zyx[0] - euler_zyx[2])
            == euler_zyx2[0] - euler_zyx2[2]
        )
        R2 = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx2)
        assert_array_almost_equal(R, R2)


def _test_conversion_matrix_euler(
    matrix_from_euler, euler_from_matrix, proper_euler
):
    """Test conversions between Euler angles and rotation matrix."""
    rng = np.random.default_rng(844)
    for _ in range(5):
        euler = (rng.random(3) - 0.5) * np.array([np.pi, 0.5 * np.pi, np.pi])
        if proper_euler:
            euler[1] += 0.5 * np.pi

        # Normal case, we can reconstruct original angles
        R = matrix_from_euler(euler)
        euler2 = euler_from_matrix(R)
        assert_array_almost_equal(euler, euler2)
        R2 = matrix_from_euler(euler2)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 1
        if proper_euler:
            euler[1] = np.pi
        else:
            euler[1] = 0.5 * np.pi
        R = matrix_from_euler(euler)
        euler2 = euler_from_matrix(R)
        assert pytest.approx(euler[1]) == euler2[1]
        R2 = matrix_from_euler(euler2)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 2
        if proper_euler:
            euler[1] = 0.0
        else:
            euler[1] = -0.5 * np.pi
        R = matrix_from_euler(euler)
        euler2 = euler_from_matrix(R)
        assert pytest.approx(euler[1]) == euler2[1]
        R2 = matrix_from_euler(euler2)
        assert_array_almost_equal(R, R2)


def test_all_euler_matrix_conversions():
    """Test all conversion between Euler angles and matrices."""
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xzx,
        pr.intrinsic_euler_xzx_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xzx,
        pr.extrinsic_euler_xzx_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xyx,
        pr.intrinsic_euler_xyx_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xyx,
        pr.extrinsic_euler_xyx_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yxy,
        pr.intrinsic_euler_yxy_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yxy,
        pr.extrinsic_euler_yxy_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yzy,
        pr.intrinsic_euler_yzy_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yzy,
        pr.extrinsic_euler_yzy_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zyz,
        pr.intrinsic_euler_zyz_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zyz,
        pr.extrinsic_euler_zyz_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zxz,
        pr.intrinsic_euler_zxz_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zxz,
        pr.extrinsic_euler_zxz_from_active_matrix,
        proper_euler=True,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xzy,
        pr.intrinsic_euler_xzy_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xzy,
        pr.extrinsic_euler_xzy_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xyz,
        pr.intrinsic_euler_xyz_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xyz,
        pr.extrinsic_euler_xyz_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yxz,
        pr.intrinsic_euler_yxz_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yxz,
        pr.extrinsic_euler_yxz_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yzx,
        pr.intrinsic_euler_yzx_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yzx,
        pr.extrinsic_euler_yzx_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zyx,
        pr.intrinsic_euler_zyx_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zyx,
        pr.extrinsic_euler_zyx_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zxy,
        pr.intrinsic_euler_zxy_from_active_matrix,
        proper_euler=False,
    )
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zxy,
        pr.extrinsic_euler_zxy_from_active_matrix,
        proper_euler=False,
    )


def test_active_matrix_from_extrinsic_roll_pitch_yaw():
    """Test conversion from extrinsic zyz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz(
            [0.5 * np.pi, 0, 0.5 * np.pi]
        ),
        np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0]
        ),
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        ),
        np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]),
    )


def test_quaternion_from_extrinsic_euler_xyz():
    """Test quaternion_from_extrinsic_euler_xyz."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        e = rng.uniform(-100, 100, [3])
        q = pr.quaternion_from_extrinsic_euler_xyz(e)
        R_from_q = pr.matrix_from_quaternion(q)
        R_from_e = pr.active_matrix_from_extrinsic_euler_xyz(e)
        assert_array_almost_equal(R_from_q, R_from_e)


def test_euler_from_quaternion_edge_case():
    quaternion = [0.57114154, -0.41689009, -0.57114154, -0.41689009]
    matrix = pr.matrix_from_quaternion(quaternion)
    euler_xyz = pr.extrinsic_euler_xyz_from_active_matrix(matrix)
    assert not np.isnan(euler_xyz).all()
