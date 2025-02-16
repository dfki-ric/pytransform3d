import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_norm_euler():
    rng = np.random.default_rng(94322)

    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
    ]
    for ea in euler_axes:
        for _ in range(10):
            e = np.pi + rng.random(3) * np.pi * 2.0
            e *= np.sign(rng.standard_normal(3))

            e_norm = pr.norm_euler(e, *ea)
            R1 = pr.matrix_from_euler(e, ea[0], ea[1], ea[2], True)
            R2 = pr.matrix_from_euler(e_norm, ea[0], ea[1], ea[2], True)
            assert_array_almost_equal(R1, R2)
            assert not np.allclose(e, e_norm)
            assert -np.pi <= e_norm[0] <= np.pi
            if ea[0] == ea[2]:
                assert 0.0 <= e_norm[1] <= np.pi
            else:
                assert -0.5 * np.pi <= e_norm[1] <= 0.5 * np.pi
            assert -np.pi <= e_norm[2] <= np.pi
            pr.assert_euler_equal(e, e_norm, *ea)


def test_euler_near_gimbal_lock():
    assert pr.euler_near_gimbal_lock([0, 0, 0], 1, 2, 1)
    assert pr.euler_near_gimbal_lock([0, -1e-7, 0], 1, 2, 1)
    assert pr.euler_near_gimbal_lock([0, 1e-7, 0], 1, 2, 1)
    assert pr.euler_near_gimbal_lock([0, np.pi, 0], 1, 2, 1)
    assert pr.euler_near_gimbal_lock([0, np.pi - 1e-7, 0], 1, 2, 1)
    assert pr.euler_near_gimbal_lock([0, np.pi + 1e-7, 0], 1, 2, 1)
    assert not pr.euler_near_gimbal_lock([0, 0.5, 0], 1, 2, 1)
    assert pr.euler_near_gimbal_lock([0, 0.5 * np.pi, 0], 0, 1, 2)
    assert pr.euler_near_gimbal_lock([0, 0.5 * np.pi - 1e-7, 0], 0, 1, 2)
    assert pr.euler_near_gimbal_lock([0, 0.5 * np.pi + 1e-7, 0], 0, 1, 2)
    assert pr.euler_near_gimbal_lock([0, -0.5 * np.pi, 0], 0, 1, 2)
    assert pr.euler_near_gimbal_lock([0, -0.5 * np.pi - 1e-7, 0], 0, 1, 2)
    assert pr.euler_near_gimbal_lock([0, -0.5 * np.pi + 1e-7, 0], 0, 1, 2)
    assert not pr.euler_near_gimbal_lock([0, 0, 0], 0, 1, 2)


def test_assert_euler_almost_equal():
    pr.assert_euler_equal(
        [0.2, 0.3, -0.5], [0.2 + np.pi, -0.3, -0.5 - np.pi], 0, 1, 0
    )
    pr.assert_euler_equal(
        [0.2, 0.3, -0.5], [0.2 + np.pi, np.pi - 0.3, -0.5 - np.pi], 0, 1, 2
    )


def test_general_matrix_euler_conversions():
    """General conversion algorithms between matrix and Euler angles."""
    rng = np.random.default_rng(22)

    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
    ]
    for ea in euler_axes:
        for extrinsic in [False, True]:
            for _ in range(5):
                e = rng.random(3)
                e[0] = 2.0 * np.pi * e[0] - np.pi
                e[1] = np.pi * e[1]
                e[2] = 2.0 * np.pi * e[2] - np.pi

                proper_euler = ea[0] == ea[2]
                if proper_euler:
                    e[1] -= np.pi / 2.0

                q = pr.quaternion_from_euler(e, ea[0], ea[1], ea[2], extrinsic)
                R = pr.matrix_from_euler(e, ea[0], ea[1], ea[2], extrinsic)
                q_R = pr.quaternion_from_matrix(R)
                pr.assert_quaternion_equal(
                    q, q_R, err_msg=f"axes: {ea}, extrinsic: {extrinsic}"
                )

                e_R = pr.euler_from_matrix(R, ea[0], ea[1], ea[2], extrinsic)
                e_q = pr.euler_from_quaternion(
                    q, ea[0], ea[1], ea[2], extrinsic
                )
                pr.assert_euler_equal(e_R, e_q, *ea)

                R_R = pr.matrix_from_euler(e_R, ea[0], ea[1], ea[2], extrinsic)
                R_q = pr.matrix_from_euler(e_q, ea[0], ea[1], ea[2], extrinsic)
                assert_array_almost_equal(R_R, R_q)


def test_from_quaternion():
    """Test conversion from quaternion to Euler angles."""
    with pytest.raises(
        ValueError, match="Axis index i \\(-1\\) must be in \\[0, 1, 2\\]"
    ):
        pr.euler_from_quaternion(pr.q_id, -1, 0, 2, True)
    with pytest.raises(
        ValueError, match="Axis index i \\(3\\) must be in \\[0, 1, 2\\]"
    ):
        pr.euler_from_quaternion(pr.q_id, 3, 0, 2, True)
    with pytest.raises(
        ValueError, match="Axis index j \\(-1\\) must be in \\[0, 1, 2\\]"
    ):
        pr.euler_from_quaternion(pr.q_id, 2, -1, 2, True)
    with pytest.raises(
        ValueError, match="Axis index j \\(3\\) must be in \\[0, 1, 2\\]"
    ):
        pr.euler_from_quaternion(pr.q_id, 2, 3, 2, True)
    with pytest.raises(
        ValueError, match="Axis index k \\(-1\\) must be in \\[0, 1, 2\\]"
    ):
        pr.euler_from_quaternion(pr.q_id, 2, 0, -1, True)
    with pytest.raises(
        ValueError, match="Axis index k \\(3\\) must be in \\[0, 1, 2\\]"
    ):
        pr.euler_from_quaternion(pr.q_id, 2, 0, 3, True)

    rng = np.random.default_rng(32)

    euler_axes = [
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 1],
        [1, 2, 1],
        [2, 1, 2],
        [2, 0, 2],
        [0, 2, 1],
        [0, 1, 2],
        [1, 0, 2],
        [1, 2, 0],
        [2, 1, 0],
        [2, 0, 1],
    ]
    functions = [
        pr.intrinsic_euler_xzx_from_active_matrix,
        pr.extrinsic_euler_xzx_from_active_matrix,
        pr.intrinsic_euler_xyx_from_active_matrix,
        pr.extrinsic_euler_xyx_from_active_matrix,
        pr.intrinsic_euler_yxy_from_active_matrix,
        pr.extrinsic_euler_yxy_from_active_matrix,
        pr.intrinsic_euler_yzy_from_active_matrix,
        pr.extrinsic_euler_yzy_from_active_matrix,
        pr.intrinsic_euler_zyz_from_active_matrix,
        pr.extrinsic_euler_zyz_from_active_matrix,
        pr.intrinsic_euler_zxz_from_active_matrix,
        pr.extrinsic_euler_zxz_from_active_matrix,
        pr.intrinsic_euler_xzy_from_active_matrix,
        pr.extrinsic_euler_xzy_from_active_matrix,
        pr.intrinsic_euler_xyz_from_active_matrix,
        pr.extrinsic_euler_xyz_from_active_matrix,
        pr.intrinsic_euler_yxz_from_active_matrix,
        pr.extrinsic_euler_yxz_from_active_matrix,
        pr.intrinsic_euler_yzx_from_active_matrix,
        pr.extrinsic_euler_yzx_from_active_matrix,
        pr.intrinsic_euler_zyx_from_active_matrix,
        pr.extrinsic_euler_zyx_from_active_matrix,
        pr.intrinsic_euler_zxy_from_active_matrix,
        pr.extrinsic_euler_zxy_from_active_matrix,
    ]
    inverse_functions = [
        pr.active_matrix_from_intrinsic_euler_xzx,
        pr.active_matrix_from_extrinsic_euler_xzx,
        pr.active_matrix_from_intrinsic_euler_xyx,
        pr.active_matrix_from_extrinsic_euler_xyx,
        pr.active_matrix_from_intrinsic_euler_yxy,
        pr.active_matrix_from_extrinsic_euler_yxy,
        pr.active_matrix_from_intrinsic_euler_yzy,
        pr.active_matrix_from_extrinsic_euler_yzy,
        pr.active_matrix_from_intrinsic_euler_zyz,
        pr.active_matrix_from_extrinsic_euler_zyz,
        pr.active_matrix_from_intrinsic_euler_zxz,
        pr.active_matrix_from_extrinsic_euler_zxz,
        pr.active_matrix_from_intrinsic_euler_xzy,
        pr.active_matrix_from_extrinsic_euler_xzy,
        pr.active_matrix_from_intrinsic_euler_xyz,
        pr.active_matrix_from_extrinsic_euler_xyz,
        pr.active_matrix_from_intrinsic_euler_yxz,
        pr.active_matrix_from_extrinsic_euler_yxz,
        pr.active_matrix_from_intrinsic_euler_yzx,
        pr.active_matrix_from_extrinsic_euler_yzx,
        pr.active_matrix_from_intrinsic_euler_zyx,
        pr.active_matrix_from_extrinsic_euler_zyx,
        pr.active_matrix_from_intrinsic_euler_zxy,
        pr.active_matrix_from_extrinsic_euler_zxy,
    ]

    fi = 0
    for ea in euler_axes:
        for extrinsic in [False, True]:
            fun = functions[fi]
            inv_fun = inverse_functions[fi]
            fi += 1
            for _ in range(5):
                e = rng.random(3)
                e[0] = 2.0 * np.pi * e[0] - np.pi
                e[1] = np.pi * e[1]
                e[2] = 2.0 * np.pi * e[2] - np.pi

                proper_euler = ea[0] == ea[2]
                if proper_euler:
                    e[1] -= np.pi / 2.0

                # normal case
                q = pr.quaternion_from_matrix(inv_fun(e))

                e1 = pr.euler_from_quaternion(q, ea[0], ea[1], ea[2], extrinsic)
                e2 = fun(pr.matrix_from_quaternion(q))
                assert_array_almost_equal(
                    e1, e2, err_msg=f"axes: {ea}, extrinsic: {extrinsic}"
                )

                # first singularity
                e[1] = 0.0
                q = pr.quaternion_from_matrix(inv_fun(e))

                R1 = inv_fun(
                    pr.euler_from_quaternion(q, ea[0], ea[1], ea[2], extrinsic)
                )
                R2 = pr.matrix_from_quaternion(q)
                assert_array_almost_equal(
                    R1, R2, err_msg=f"axes: {ea}, extrinsic: {extrinsic}"
                )

                # second singularity
                e[1] = np.pi
                q = pr.quaternion_from_matrix(inv_fun(e))

                R1 = inv_fun(
                    pr.euler_from_quaternion(q, ea[0], ea[1], ea[2], extrinsic)
                )
                R2 = pr.matrix_from_quaternion(q)
                assert_array_almost_equal(
                    R1, R2, err_msg=f"axes: {ea}, extrinsic: {extrinsic}"
                )
