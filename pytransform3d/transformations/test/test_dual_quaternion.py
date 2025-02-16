import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def test_check_dual_quaternion():
    dq1 = [0, 0]
    with pytest.raises(ValueError, match="Expected dual quaternion"):
        pt.check_dual_quaternion(dq1)

    dq2 = [[0, 0]]
    with pytest.raises(ValueError, match="Expected dual quaternion"):
        pt.check_dual_quaternion(dq2)

    dq3 = pt.check_dual_quaternion([0] * 8, unit=False)
    assert dq3.shape[0] == 8
    assert pt.dual_quaternion_requires_renormalization(dq3)
    dq4 = pt.check_dual_quaternion(dq3, unit=True)
    assert not pt.dual_quaternion_requires_renormalization(dq4)

    dq5 = np.array(
        [
            0.94508498,
            0.0617101,
            -0.06483886,
            0.31432811,
            -0.07743254,
            0.04985168,
            -0.26119618,
            0.1691491,
        ]
    )
    dq5_not_orthogonal = np.copy(dq5)
    dq5_not_orthogonal[4:] = np.round(dq5_not_orthogonal[4:], 1)
    assert pt.dual_quaternion_requires_renormalization(dq5_not_orthogonal)


def test_normalize_dual_quaternion():
    dq = [1, 0, 0, 0, 0, 0, 0, 0]
    dq_norm = pt.check_dual_quaternion(dq)
    pt.assert_unit_dual_quaternion(dq_norm)
    assert_array_almost_equal(dq, dq_norm)
    assert_array_almost_equal(dq, pt.norm_dual_quaternion(dq))

    dq = [0, 0, 0, 0, 0, 0, 0, 0]
    dq_norm = pt.check_dual_quaternion(dq)
    pt.assert_unit_dual_quaternion(dq_norm)
    assert_array_almost_equal([1, 0, 0, 0, 0, 0, 0, 0], dq_norm)
    assert_array_almost_equal(dq_norm, pt.norm_dual_quaternion(dq))

    dq = [0, 0, 0, 0, 0.3, 0.5, 0, 0.2]
    dq_norm = pt.check_dual_quaternion(dq)
    assert pt.dual_quaternion_requires_renormalization(dq_norm)
    assert_array_almost_equal([1, 0, 0, 0, 0.3, 0.5, 0, 0.2], dq_norm)
    assert not pt.dual_quaternion_requires_renormalization(
        pt.norm_dual_quaternion(dq)
    )

    rng = np.random.default_rng(999)
    for _ in range(5):  # norm != 1
        A2B = pt.random_transform(rng)
        dq = rng.standard_normal() * pt.dual_quaternion_from_transform(A2B)
        dq_norm = pt.check_dual_quaternion(dq)
        pt.assert_unit_dual_quaternion(dq_norm)
        assert_array_almost_equal(dq_norm, pt.norm_dual_quaternion(dq))

    for _ in range(5):  # real and dual quaternion are not orthogonal
        A2B = pt.random_transform(rng)
        dq = pt.dual_quaternion_from_transform(A2B)
        dq_roundoff_error = np.copy(dq)
        dq_roundoff_error[4:] = np.round(dq_roundoff_error[4:], 3)
        assert pt.dual_quaternion_requires_renormalization(dq_roundoff_error)
        dq_norm = pt.norm_dual_quaternion(dq_roundoff_error)
        pt.assert_unit_dual_quaternion(dq_norm)
        assert not pt.dual_quaternion_requires_renormalization(dq_norm)
        assert_array_almost_equal(dq, dq_norm, decimal=3)


def test_dual_quaternion_double():
    rng = np.random.default_rng(4183)
    A2B = pt.random_transform(rng)
    dq = pt.dual_quaternion_from_transform(A2B)
    dq_double = pt.dual_quaternion_double(dq)
    pt.assert_unit_dual_quaternion_equal(dq, dq_double)
    assert_array_almost_equal(A2B, pt.transform_from_dual_quaternion(dq_double))


def test_dual_quaternion_concatenation():
    rng = np.random.default_rng(1000)
    for _ in range(5):
        A2B = pt.random_transform(rng)
        B2C = pt.random_transform(rng)
        A2C = pt.concat(A2B, B2C)
        dq1 = pt.dual_quaternion_from_transform(A2B)
        dq2 = pt.dual_quaternion_from_transform(B2C)
        dq3 = pt.concatenate_dual_quaternions(dq2, dq1)
        A2C2 = pt.transform_from_dual_quaternion(dq3)
        assert_array_almost_equal(A2C, A2C2)


def test_dual_quaternion_applied_to_point():
    rng = np.random.default_rng(1000)
    for _ in range(5):
        p_A = pr.random_vector(rng, 3)
        A2B = pt.random_transform(rng)
        dq = pt.dual_quaternion_from_transform(A2B)
        p_B = pt.dq_prod_vector(dq, p_A)
        assert_array_almost_equal(
            p_B, pt.transform(A2B, pt.vector_to_point(p_A))[:3]
        )


def test_dual_quaternion_sclerp_same_dual_quaternions():
    rng = np.random.default_rng(19)
    pose = pt.random_transform(rng)
    dq = pt.dual_quaternion_from_transform(pose)
    dq2 = pt.dual_quaternion_sclerp(dq, dq, 0.5)
    assert_array_almost_equal(dq, dq2)


def test_dual_quaternion_sclerp():
    rng = np.random.default_rng(22)
    pose1 = pt.random_transform(rng)
    pose2 = pt.random_transform(rng)
    dq1 = pt.dual_quaternion_from_transform(pose1)
    dq2 = pt.dual_quaternion_from_transform(pose2)

    n_steps = 100

    # Ground truth: screw linear interpolation
    pose12pose2 = pt.concat(pose2, pt.invert_transform(pose1))
    Stheta = pt.exponential_coordinates_from_transform(pose12pose2)
    offsets = np.array(
        [
            pt.transform_from_exponential_coordinates(Stheta * t)
            for t in np.linspace(0, 1, n_steps)
        ]
    )
    interpolated_poses = np.array(
        [pt.concat(offset, pose1) for offset in offsets]
    )

    # Dual quaternion ScLERP
    sclerp_interpolated_dqs = np.vstack(
        [
            pt.dual_quaternion_sclerp(dq1, dq2, t)
            for t in np.linspace(0, 1, n_steps)
        ]
    )
    sclerp_interpolated_poses_from_dqs = np.array(
        [
            pt.transform_from_dual_quaternion(dq)
            for dq in sclerp_interpolated_dqs
        ]
    )

    # Transformation matrix ScLERP
    sclerp_interpolated_transforms = np.array(
        [
            pt.transform_sclerp(pose1, pose2, t)
            for t in np.linspace(0, 1, n_steps)
        ]
    )

    for t in range(n_steps):
        assert_array_almost_equal(
            interpolated_poses[t], sclerp_interpolated_poses_from_dqs[t]
        )
        assert_array_almost_equal(
            interpolated_poses[t], sclerp_interpolated_transforms[t]
        )


def test_dual_quaternion_sclerp_sign_ambiguity():
    n_steps = 10
    rng = np.random.default_rng(2323)
    T1 = pt.random_transform(rng)
    dq1 = pt.dual_quaternion_from_transform(T1)
    dq2 = np.copy(dq1)

    if np.sign(dq1[0]) != np.sign(dq2[0]):
        dq2 *= -1.0
    traj_q = [
        pt.dual_quaternion_sclerp(dq1, dq2, t)
        for t in np.linspace(0, 1, n_steps)
    ]
    path_length = np.sum(
        [np.linalg.norm(r - s) for r, s in zip(traj_q[:-1], traj_q[1:])]
    )

    dq2 *= -1.0
    traj_q_opposing = [
        pt.dual_quaternion_sclerp(dq1, dq2, t)
        for t in np.linspace(0, 1, n_steps)
    ]
    path_length_opposing = np.sum(
        [
            np.linalg.norm(r - s)
            for r, s in zip(traj_q_opposing[:-1], traj_q_opposing[1:])
        ]
    )

    assert path_length_opposing == path_length


def test_compare_dual_quaternion_and_transform_power():
    rng = np.random.default_rng(44329)
    for _ in range(20):
        t = rng.normal()
        A2B = pt.random_transform(rng)
        dq = pt.dual_quaternion_from_transform(A2B)
        assert_array_almost_equal(
            pt.transform_power(A2B, t),
            pt.transform_from_dual_quaternion(pt.dual_quaternion_power(dq, t)),
        )


def test_screw_parameters_from_dual_quaternion():
    dq = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    q, s_axis, h, theta = pt.screw_parameters_from_dual_quaternion(dq)
    assert_array_almost_equal(q, np.zeros(3))
    assert_array_almost_equal(s_axis, np.array([1, 0, 0]))
    assert np.isinf(h)
    assert pytest.approx(theta) == 0

    dq = pt.dual_quaternion_from_pq(np.array([1.2, 1.3, 1.4, 1, 0, 0, 0]))
    q, s_axis, h, theta = pt.screw_parameters_from_dual_quaternion(dq)
    assert_array_almost_equal(q, np.zeros(3))
    assert_array_almost_equal(s_axis, pr.norm_vector(np.array([1.2, 1.3, 1.4])))
    assert np.isinf(h)
    assert pytest.approx(theta) == np.linalg.norm(np.array([1.2, 1.3, 1.4]))

    rng = np.random.default_rng(1001)
    quat = pr.random_quaternion(rng)
    a = pr.axis_angle_from_quaternion(quat)
    dq = pt.dual_quaternion_from_pq(np.r_[0, 0, 0, quat])
    q, s_axis, h, theta = pt.screw_parameters_from_dual_quaternion(dq)
    assert_array_almost_equal(q, np.zeros(3))
    assert_array_almost_equal(s_axis, a[:3])
    assert_array_almost_equal(h, 0)
    assert_array_almost_equal(theta, a[3])

    for _ in range(5):
        A2B = pt.random_transform(rng)
        dq = pt.dual_quaternion_from_transform(A2B)
        Stheta = pt.exponential_coordinates_from_transform(A2B)
        S, theta = pt.screw_axis_from_exponential_coordinates(Stheta)
        q, s_axis, h = pt.screw_parameters_from_screw_axis(S)
        q2, s_axis2, h2, theta2 = pt.screw_parameters_from_dual_quaternion(dq)
        pt.assert_screw_parameters_equal(
            q, s_axis, h, theta, q2, s_axis2, h2, theta2
        )
