import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal, assert_true
from pytransform.rotations import *


def test_norm_vector():
    """Test normalization of vectors."""
    random_state = np.random.RandomState(0)
    for n in range(1, 6):
        v = random_vector(random_state, n)
        u = norm_vector(v)
        assert_almost_equal(np.linalg.norm(u), 1)


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = norm_vector(np.zeros(3))
    assert_true(np.isfinite(np.linalg.norm(normalized)))


def test_norm_axis_angle():
    """Test normalization of angle-axis representation."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        angle = random_state.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a[3] = angle
        n = norm_axis_angle(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a[3] = angle + angle_offset
            n2 = norm_axis_angle(a)
            assert_array_almost_equal(n, n2)


def test_perpendicular_to_vectors():
    """Test function to compute perpendicular to vectors."""
    random_state = np.random.RandomState(0)
    a = norm_vector(random_vector(random_state))
    a1 = norm_vector(random_vector(random_state))
    b = norm_vector(perpendicular_to_vectors(a, a1))
    c = norm_vector(perpendicular_to_vectors(a, b))
    assert_almost_equal(angle_between_vectors(a, b), np.pi / 2.0)
    assert_almost_equal(angle_between_vectors(a, c), np.pi / 2.0)
    assert_almost_equal(angle_between_vectors(b, c), np.pi / 2.0)
    assert_array_almost_equal(perpendicular_to_vectors(b, c), a)
    assert_array_almost_equal(perpendicular_to_vectors(c, a), b)


def test_angle_between_vectors():
    """Test function to compute angle between two vectors."""
    v = np.array([1, 0, 0])
    a = np.array([0, 1, 0, np.pi / 2])
    R = matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert_almost_equal(angle_between_vectors(vR, v), a[-1])
    v = np.array([0, 1, 0])
    a = np.array([1, 0, 0, np.pi / 2])
    R = matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert_almost_equal(angle_between_vectors(vR, v), a[-1])
    v = np.array([0, 0, 1])
    a = np.array([1, 0, 0, np.pi / 2])
    R = matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert_almost_equal(angle_between_vectors(vR, v), a[-1])


def test_cross_product_matrix():
    """Test cross-product matrix."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        v = random_vector(random_state)
        w = random_vector(random_state)
        V = cross_product_matrix(v)
        r1 = np.cross(v, w)
        r2 = np.dot(V, w)
        assert_array_almost_equal(r1, r2)


def test_conversions_matrix_euler_xyz():
    """Test conversions between rotation matrix and xyz Euler angles."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        euler_xyz = euler_xyz_from_matrix(R)
        R2 = matrix_from_euler_xyz(euler_xyz)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)

        euler_xyz2 = euler_xyz_from_matrix(R2)
        assert_array_almost_equal(euler_xyz, euler_xyz2)


def test_conversions_matrix_euler_zyx():
    """Test conversions between rotation matrix and zyx Euler angles."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        euler_zyx = euler_zyx_from_matrix(R)
        R2 = matrix_from_euler_zyx(euler_zyx)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)

        euler_zyx2 = euler_zyx_from_matrix(R2)
        assert_array_almost_equal(euler_zyx, euler_zyx2)


def test_conversions_matrix_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        a2 = axis_angle_from_matrix(R)
        assert_array_almost_equal(a, a2)

        R2 = matrix_from_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)


def test_conversions_matrix_quaternion():
    """Test conversions between rotation matrix and quaternion."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = random_quaternion(random_state)
        R = matrix_from_quaternion(q)
        assert_rotation_matrix(R)

        q2 = quaternion_from_matrix(R)
        assert_quaternion_equal(q, q2)

        R2 = matrix_from_quaternion(q2)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)


def test_conversions_axis_angle_quaternion():
    """Test conversions between axis-angle and quaternion."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        q = quaternion_from_axis_angle(a)

        a2 = axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = quaternion_from_axis_angle(a2)
        assert_quaternion_equal(q, q2)


def test_interpolate_axis_angle():
    """Test interpolation between two axis-angle rotations with slerp."""
    n_steps = 10
    random_state = np.random.RandomState(1)
    a1 = random_axis_angle(random_state)
    a2 = random_axis_angle(random_state)

    traj = [axis_angle_slerp(a1, a2, t) for t in np.linspace(0, 1, n_steps)]

    axis = norm_vector(perpendicular_to_vectors(a1[:3], a2[:3]))
    angle = angle_between_vectors(a1[:3], a2[:3])
    traj2 = []
    for t in np.linspace(0, 1, n_steps):
        inta = np.hstack((axis, (t * angle,)))
        intaxis = matrix_from_axis_angle(inta).dot(a1[:3])
        intangle = (1 - t) * a1[3] + t * a2[3]
        traj2.append(np.hstack((intaxis, (intangle,))))

    assert_array_almost_equal(traj, traj2)


def test_interpolate_quaternion():
    """Test interpolation between two quaternions with slerp."""
    n_steps = 10
    random_state = np.random.RandomState(0)
    a1 = random_axis_angle(random_state)
    a2 = random_axis_angle(random_state)
    q1 = quaternion_from_axis_angle(a1)
    q2 = quaternion_from_axis_angle(a2)

    traj_q = [quaternion_slerp(q1, q2, t) for t in np.linspace(0, 1, n_steps)]
    traj_R = [matrix_from_quaternion(q) for q in traj_q]
    R_diff = np.diff(traj_R, axis=0)
    R_diff_norms = [np.linalg.norm(Rd) for Rd in R_diff]
    assert_array_almost_equal(R_diff_norms,
                              R_diff_norms[0] * np.ones(n_steps - 1))


def test_concatenate_quaternions():
    """Test concatenation of to quaternions."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q1 = quaternion_from_axis_angle(random_axis_angle(random_state))
        q2 = quaternion_from_axis_angle(random_axis_angle(random_state))

        R1 = matrix_from_quaternion(q1)
        R2 = matrix_from_quaternion(q2)

        q12 = concatenate_quaternions(q1, q2)
        R12 = np.dot(R1, R2)
        q12R = quaternion_from_matrix(R12)

        assert_quaternion_equal(q12, q12R)


def test_quaternion_rotation():
    """Test quaternion rotation."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = quaternion_from_axis_angle(random_axis_angle(random_state))
        R = matrix_from_quaternion(q)
        v = random_vector(random_state)
        vR = np.dot(R, v)
        vq = q_prod_vector(q, v)
        assert_array_almost_equal(vR, vq)


def test_quaternion_conjugate():
    """Test quaternion conjugate."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = quaternion_from_axis_angle(random_axis_angle(random_state))
        v = random_vector(random_state)
        vq = q_prod_vector(q, v)
        vq2 = concatenate_quaternions(concatenate_quaternions(
            q, np.hstack(([0], v))), q_conj(q))[1:]
        assert_array_almost_equal(vq, vq2)


def test_quaternion_dist():
    """Test angular metric of quaternions."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q1 = quaternion_from_axis_angle(random_axis_angle(random_state))
        q2 = quaternion_from_axis_angle(random_axis_angle(random_state))
        q1_to_q1 = quaternion_dist(q1, q1)
        if q1_to_q1 > np.pi:
            q1_to_q1 = 2.0 * np.pi - q1_to_q1
        assert_almost_equal(q1_to_q1, 0.0)
        q2_to_q2 = quaternion_dist(q2, q2)
        if q2_to_q2 > np.pi:
            q2_to_q2 = 2.0 * np.pi - q2_to_q2
        assert_almost_equal(q2_to_q2, 0.0)
        q1_to_q2 = quaternion_dist(q1, q2)
        q2_to_q1 = quaternion_dist(q2, q1)
        assert_almost_equal(q1_to_q2, q2_to_q1)
