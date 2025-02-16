import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_q_R_slerps():
    """Compare SLERP implementations for quaternions and rotation matrices."""
    rng = np.random.default_rng(833234)
    for _ in range(20):
        q_start = pr.random_quaternion(rng)
        q_end = pr.random_quaternion(rng)
        R_start, R_end = (
            pr.matrix_from_quaternion(q_start),
            pr.matrix_from_quaternion(q_end),
        )
        t = rng.random()
        q_t = pr.quaternion_slerp(q_start, q_end, t, shortest_path=True)
        R_t = pr.matrix_slerp(R_start, R_end, t)
        assert_array_almost_equal(R_t, pr.matrix_from_quaternion(q_t))


def test_rotation_matrix_power():
    """Test power of rotation matrices."""
    R = pr.random_matrix(rng=np.random.default_rng(2844))
    angle = pr.axis_angle_from_matrix(R)[-1]

    R_m2 = pr.matrix_power(R, -2.0)
    assert_array_almost_equal(R_m2, R.T.dot(R.T))

    R_m1 = pr.matrix_power(R, -1.0)
    assert_array_almost_equal(R_m1, R.T)

    R_m05 = pr.matrix_power(R, -0.5)
    assert pytest.approx(angle) == 2 * pr.axis_angle_from_matrix(R_m05)[-1]

    R_0 = pr.matrix_power(R, 0.0)
    assert_array_almost_equal(R_0, np.eye(3))

    R_p05 = pr.matrix_power(R, 0.5)
    assert pytest.approx(angle) == 2 * pr.axis_angle_from_matrix(R_p05)[-1]

    R_p1 = pr.matrix_power(R, 1.0)
    assert_array_almost_equal(R_p1, R)

    R_p2 = pr.matrix_power(R, 2.0)
    assert_array_almost_equal(R_p2, R.dot(R))


def test_interpolate_axis_angle():
    """Test interpolation between two axis-angle rotations with slerp."""
    n_steps = 10
    rng = np.random.default_rng(1)
    a1 = pr.random_axis_angle(rng)
    a2 = pr.random_axis_angle(rng)

    traj = [pr.axis_angle_slerp(a1, a2, t) for t in np.linspace(0, 1, n_steps)]

    axis = pr.norm_vector(pr.perpendicular_to_vectors(a1[:3], a2[:3]))
    angle = pr.angle_between_vectors(a1[:3], a2[:3])
    traj2 = []
    for t in np.linspace(0, 1, n_steps):
        inta = np.hstack((axis, (t * angle,)))
        intaxis = pr.matrix_from_axis_angle(inta).dot(a1[:3])
        intangle = (1 - t) * a1[3] + t * a2[3]
        traj2.append(np.hstack((intaxis, (intangle,))))

    assert_array_almost_equal(traj, traj2)


def test_interpolate_same_axis_angle():
    """Test interpolation between the same axis-angle rotation.

    See issue #45.
    """
    n_steps = 3
    rng = np.random.default_rng(42)
    a = pr.random_axis_angle(rng)
    traj = [pr.axis_angle_slerp(a, a, t) for t in np.linspace(0, 1, n_steps)]
    assert len(traj) == n_steps
    assert_array_almost_equal(traj[0], a)
    assert_array_almost_equal(traj[1], a)
    assert_array_almost_equal(traj[2], a)


def test_interpolate_almost_same_axis_angle():
    """Test interpolation between almost the same axis-angle rotation."""
    n_steps = 3
    rng = np.random.default_rng(42)
    a1 = pr.random_axis_angle(rng)
    a2 = np.copy(a1)
    a2[-1] += np.finfo("float").eps
    traj = [pr.axis_angle_slerp(a1, a2, t) for t in np.linspace(0, 1, n_steps)]
    assert len(traj) == n_steps
    assert_array_almost_equal(traj[0], a1)
    assert_array_almost_equal(traj[1], a1)
    assert_array_almost_equal(traj[2], a2)


def test_interpolate_quaternion():
    """Test interpolation between two quaternions with slerp."""
    n_steps = 10
    rng = np.random.default_rng(0)
    a1 = pr.random_axis_angle(rng)
    a2 = pr.random_axis_angle(rng)
    q1 = pr.quaternion_from_axis_angle(a1)
    q2 = pr.quaternion_from_axis_angle(a2)

    traj_q = [
        pr.quaternion_slerp(q1, q2, t) for t in np.linspace(0, 1, n_steps)
    ]
    traj_R = [pr.matrix_from_quaternion(q) for q in traj_q]
    R_diff = np.diff(traj_R, axis=0)
    R_diff_norms = [np.linalg.norm(Rd) for Rd in R_diff]
    assert_array_almost_equal(
        R_diff_norms, R_diff_norms[0] * np.ones(n_steps - 1)
    )


def test_interpolate_quaternion_shortest_path():
    """Test SLERP between similar quternions with opposite sign."""
    n_steps = 10
    rng = np.random.default_rng(2323)
    q1 = pr.random_quaternion(rng)
    a1 = pr.axis_angle_from_quaternion(q1)
    a2 = np.r_[a1[:3], a1[3] * 1.1]
    q2 = pr.quaternion_from_axis_angle(a2)

    if np.sign(q1[0]) != np.sign(q2[0]):
        q2 *= -1.0
    traj_q = [
        pr.quaternion_slerp(q1, q2, t) for t in np.linspace(0, 1, n_steps)
    ]
    path_length = np.sum(
        [pr.quaternion_dist(r, s) for r, s in zip(traj_q[:-1], traj_q[1:])]
    )

    q2 *= -1.0
    traj_q_opposing = [
        pr.quaternion_slerp(q1, q2, t) for t in np.linspace(0, 1, n_steps)
    ]
    path_length_opposing = np.sum(
        [
            pr.quaternion_dist(r, s)
            for r, s in zip(traj_q_opposing[:-1], traj_q_opposing[1:])
        ]
    )

    assert path_length_opposing > path_length

    traj_q_opposing_corrected = [
        pr.quaternion_slerp(q1, q2, t, shortest_path=True)
        for t in np.linspace(0, 1, n_steps)
    ]
    path_length_opposing_corrected = np.sum(
        [
            pr.quaternion_dist(r, s)
            for r, s in zip(
                traj_q_opposing_corrected[:-1], traj_q_opposing_corrected[1:]
            )
        ]
    )

    assert pytest.approx(path_length_opposing_corrected) == path_length


def test_interpolate_same_quaternion():
    """Test interpolation between the same quaternion rotation.

    See issue #45.
    """
    n_steps = 3
    rng = np.random.default_rng(42)
    a = pr.random_axis_angle(rng)
    q = pr.quaternion_from_axis_angle(a)
    traj = [pr.quaternion_slerp(q, q, t) for t in np.linspace(0, 1, n_steps)]
    assert len(traj) == n_steps
    assert_array_almost_equal(traj[0], q)
    assert_array_almost_equal(traj[1], q)
    assert_array_almost_equal(traj[2], q)


def test_interpolate_shortest_path_same_quaternion():
    """Test interpolate along shortest path with same quaternion."""
    rng = np.random.default_rng(8353)
    q = pr.random_quaternion(rng)
    q_interpolated = pr.quaternion_slerp(q, q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)

    q = np.array([0.0, 1.0, 0.0, 0.0])
    q_interpolated = pr.quaternion_slerp(q, -q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)

    q = np.array([0.0, 0.0, 1.0, 0.0])
    q_interpolated = pr.quaternion_slerp(q, -q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)

    q = np.array([0.0, 0.0, 0.0, 1.0])
    q_interpolated = pr.quaternion_slerp(q, -q, 0.5, shortest_path=True)
    assert_array_almost_equal(q, q_interpolated)


def test_rotor_slerp():
    rng = np.random.default_rng(86)
    for _ in range(5):
        a_unit = pr.norm_vector(rng.standard_normal(3))
        b_unit = pr.norm_vector(rng.standard_normal(3))
        rotor1 = pr.rotor_from_two_directions(a_unit, b_unit)

        axis = pr.norm_vector(np.cross(a_unit, b_unit))
        angle = pr.angle_between_vectors(a_unit, b_unit)
        q1 = pr.quaternion_from_axis_angle(np.r_[axis, angle])

        c_unit = pr.norm_vector(rng.standard_normal(3))
        d_unit = pr.norm_vector(rng.standard_normal(3))
        rotor2 = pr.rotor_from_two_directions(c_unit, d_unit)

        axis = pr.norm_vector(np.cross(c_unit, d_unit))
        angle = pr.angle_between_vectors(c_unit, d_unit)
        q2 = pr.quaternion_from_axis_angle(np.r_[axis, angle])

        rotor_025 = pr.rotor_slerp(rotor1, rotor2, 0.25)
        q_025 = pr.quaternion_slerp(q1, q2, 0.25)

        e = rng.standard_normal(3)
        assert_array_almost_equal(
            pr.rotor_apply(rotor_025, e), pr.q_prod_vector(q_025, e)
        )

        rotor_075 = pr.rotor_slerp(rotor1, rotor2, 0.25)
        q_075 = pr.quaternion_slerp(q1, q2, 0.25)

        e = rng.standard_normal(3)
        assert_array_almost_equal(
            pr.rotor_apply(rotor_075, e), pr.q_prod_vector(q_075, e)
        )
