import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import (assert_almost_equal, assert_equal, assert_true,
                        assert_greater, assert_raises_regexp, assert_less,
                        assert_false)

import pytransform3d.rotations as pr


def test_norm_vector():
    """Test normalization of vectors."""
    random_state = np.random.RandomState(0)
    for n in range(1, 6):
        v = pr.random_vector(random_state, n)
        u = pr.norm_vector(v)
        assert_almost_equal(np.linalg.norm(u), 1)


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = pr.norm_vector(np.zeros(3))
    assert_true(np.isfinite(np.linalg.norm(normalized)))


def test_norm_angle():
    """Test normalization of angle."""
    random_state = np.random.RandomState(0)
    a_norm = random_state.uniform(-np.pi, np.pi, size=(100,))
    for b in np.linspace(-10.0 * np.pi, 10.0 * np.pi, 11):
        a = a_norm + b
        assert_array_almost_equal(pr.norm_angle(a), a_norm)

    assert_almost_equal(pr.norm_angle(-np.pi), np.pi)
    assert_almost_equal(pr.norm_angle(np.pi), np.pi)


def test_norm_axis_angle():
    """Test normalization of angle-axis representation."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    n = pr.norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    n = pr.norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    n = pr.norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        angle = random_state.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a[3] = angle
        n = pr.norm_axis_angle(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a[3] = angle + angle_offset
            n2 = pr.norm_axis_angle(a)
            assert_array_almost_equal(n, n2)


def test_norm_compact_axis_angle():
    """Test normalization of compact angle-axis representation."""
    a = np.array([np.pi, 0.0, 0.0])
    n = pr.norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, np.pi, 0.0])
    n = pr.norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, np.pi])
    n = pr.norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_compact_axis_angle(random_state)
        axis = a / np.linalg.norm(a)
        angle = random_state.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a = axis * angle
        n = pr.norm_compact_axis_angle(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a = axis * (angle + angle_offset)
            n2 = pr.norm_compact_axis_angle(a)
            assert_array_almost_equal(n, n2)


def test_perpendicular_to_vectors():
    """Test function to compute perpendicular to vectors."""
    random_state = np.random.RandomState(0)
    a = pr.norm_vector(pr.random_vector(random_state))
    a1 = pr.norm_vector(pr.random_vector(random_state))
    b = pr.norm_vector(pr.perpendicular_to_vectors(a, a1))
    c = pr.norm_vector(pr.perpendicular_to_vectors(a, b))
    assert_almost_equal(pr.angle_between_vectors(a, b), np.pi / 2.0)
    assert_almost_equal(pr.angle_between_vectors(a, c), np.pi / 2.0)
    assert_almost_equal(pr.angle_between_vectors(b, c), np.pi / 2.0)
    assert_array_almost_equal(pr.perpendicular_to_vectors(b, c), a)
    assert_array_almost_equal(pr.perpendicular_to_vectors(c, a), b)


def test_perpendicular_to_vector():
    """Test function to compute perpendicular to vector."""
    assert_almost_equal(pr.angle_between_vectors(
        pr.unitx, pr.perpendicular_to_vector(pr.unitx)), np.pi / 2.0)
    assert_almost_equal(pr.angle_between_vectors(
        pr.unity, pr.perpendicular_to_vector(pr.unity)), np.pi / 2.0)
    assert_almost_equal(pr.angle_between_vectors(
        pr.unitz, pr.perpendicular_to_vector(pr.unitz)), np.pi / 2.0)
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.norm_vector(pr.random_vector(random_state))
        assert_almost_equal(pr.angle_between_vectors(
            a, pr.perpendicular_to_vector(a)), np.pi / 2.0)
        b = a - np.array([a[0], 0.0, 0.0])
        assert_almost_equal(pr.angle_between_vectors(
            b, pr.perpendicular_to_vector(b)), np.pi / 2.0)
        c = a - np.array([0.0, a[1], 0.0])
        assert_almost_equal(pr.angle_between_vectors(
            c, pr.perpendicular_to_vector(c)), np.pi / 2.0)
        d = a - np.array([0.0, 0.0, a[2]])
        assert_almost_equal(pr.angle_between_vectors(
            d, pr.perpendicular_to_vector(d)), np.pi / 2.0)


def test_angle_between_vectors():
    """Test function to compute angle between two vectors."""
    v = np.array([1, 0, 0])
    a = np.array([0, 1, 0, np.pi / 2])
    R = pr.matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert_almost_equal(pr.angle_between_vectors(vR, v), a[-1])
    v = np.array([0, 1, 0])
    a = np.array([1, 0, 0, np.pi / 2])
    R = pr.matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert_almost_equal(pr.angle_between_vectors(vR, v), a[-1])
    v = np.array([0, 0, 1])
    a = np.array([1, 0, 0, np.pi / 2])
    R = pr.matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert_almost_equal(pr.angle_between_vectors(vR, v), a[-1])


def test_angle_between_close_vectors():
    """Test angle between close vectors.

    See issue #47.
    """
    a = np.array([0.9689124217106448, 0.24740395925452294, 0.0, 0.0])
    b = np.array([0.9689124217106448, 0.247403959254523, 0.0, 0.0])
    angle = pr.angle_between_vectors(a, b)
    assert_almost_equal(angle, 0.0)


def test_angle_to_zero_vector_is_nan():
    """Test angle to zero vector."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    with warnings.catch_warnings(record=True) as w:
        angle = pr.angle_between_vectors(a, b)
        assert_equal(len(w), 1)
    assert_true(np.isnan(angle))


def test_vector_projection_on_zero_vector():
    """Test projection on zero vector."""
    random_state = np.random.RandomState(23)
    for _ in range(5):
        a = pr.random_vector(random_state, 3)
        a_on_b = pr.vector_projection(a, np.zeros(3))
        assert_array_almost_equal(a_on_b, np.zeros(3))


def test_vector_projection():
    """Test orthogonal projection of one vector to another vector."""
    a = np.ones(3)
    a_on_unitx = pr.vector_projection(a, pr.unitx)
    assert_array_almost_equal(a_on_unitx, pr.unitx)
    assert_almost_equal(pr.angle_between_vectors(a_on_unitx, pr.unitx), 0.0)

    a2_on_unitx = pr.vector_projection(2 * a, pr.unitx)
    assert_array_almost_equal(a2_on_unitx, 2 * pr.unitx)
    assert_almost_equal(pr.angle_between_vectors(a2_on_unitx, pr.unitx), 0.0)

    a_on_unity = pr.vector_projection(a, pr.unity)
    assert_array_almost_equal(a_on_unity, pr.unity)
    assert_almost_equal(pr.angle_between_vectors(a_on_unity, pr.unity), 0.0)

    minus_a_on_unity = pr.vector_projection(-a, pr.unity)
    assert_array_almost_equal(minus_a_on_unity, -pr.unity)
    assert_almost_equal(
        pr.angle_between_vectors(minus_a_on_unity, pr.unity), np.pi)

    a_on_unitz = pr.vector_projection(a, pr.unitz)
    assert_array_almost_equal(a_on_unitz, pr.unitz)
    assert_almost_equal(pr.angle_between_vectors(a_on_unitz, pr.unitz), 0.0)

    unitz_on_a = pr.vector_projection(pr.unitz, a)
    assert_array_almost_equal(unitz_on_a, np.ones(3) / 3.0)
    assert_almost_equal(pr.angle_between_vectors(unitz_on_a, a), 0.0)

    unitx_on_unitx = pr.vector_projection(pr.unitx, pr.unitx)
    assert_array_almost_equal(unitx_on_unitx, pr.unitx)
    assert_almost_equal(
        pr.angle_between_vectors(unitx_on_unitx, pr.unitx), 0.0)


def test_check_skew_symmetric_matrix():
    assert_raises_regexp(
        ValueError, "Expected skew-symmetric matrix with shape",
        pr.check_skew_symmetric_matrix, [])
    assert_raises_regexp(
        ValueError, "Expected skew-symmetric matrix with shape",
        pr.check_skew_symmetric_matrix, np.zeros((3, 4)))
    assert_raises_regexp(
        ValueError, "Expected skew-symmetric matrix with shape",
        pr.check_skew_symmetric_matrix, np.zeros((4, 3)))
    V = np.zeros((3, 3))
    V[0, 0] = 0.001
    assert_raises_regexp(
        ValueError, "Expected skew-symmetric matrix, but it failed the test",
        pr.check_skew_symmetric_matrix, V)
    with warnings.catch_warnings(record=True) as w:
        pr.check_skew_symmetric_matrix(V, strict_check=False)
        assert_equal(len(w), 1)

    pr.check_skew_symmetric_matrix(np.zeros((3, 3)))


def test_cross_product_matrix():
    """Test cross-product matrix."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        v = pr.random_vector(random_state)
        w = pr.random_vector(random_state)
        V = pr.cross_product_matrix(v)
        pr.check_skew_symmetric_matrix(V)
        r1 = np.cross(v, w)
        r2 = np.dot(V, w)
        assert_array_almost_equal(r1, r2)


def test_check_matrix():
    """Test input validation for rotation matrix."""
    R_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    R = pr.check_matrix(R_list)
    assert_equal(type(R), np.ndarray)
    assert_equal(R.dtype, np.float64)

    R_int_array = np.eye(3, dtype=int)
    R = pr.check_matrix(R_int_array)
    assert_equal(type(R), np.ndarray)
    assert_equal(R.dtype, np.float64)

    R_array = np.eye(3)
    R = pr.check_matrix(R_array)
    assert_array_equal(R_array, R)

    R = np.eye(4)
    assert_raises_regexp(
        ValueError, "Expected rotation matrix with shape",
        pr.check_matrix, R)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0.1, 1]])
    assert_raises_regexp(
        ValueError, "inversion by transposition", pr.check_matrix, R)

    R = np.array([[1, 0, 1e-16], [0, 1, 0], [0, 0, 1]])
    R2 = pr.check_matrix(R)
    assert_array_equal(R, R2)

    R = -np.eye(3)
    assert_raises_regexp(ValueError, "determinant", pr.check_matrix, R)


def test_check_axis_angle():
    """Test input validation for axis-angle representation."""
    a_list = [1, 0, 0, 0]
    a = pr.check_axis_angle(a_list)
    assert_array_almost_equal(a_list, a)
    assert_equal(type(a), np.ndarray)
    assert_equal(a.dtype, np.float64)

    random_state = np.random.RandomState(0)
    a = np.empty(4)
    a[:3] = pr.random_vector(random_state, 3)
    a[3] = random_state.randn() * 4.0 * np.pi
    a2 = pr.check_axis_angle(a)
    pr.assert_axis_angle_equal(a, a2)
    assert_almost_equal(np.linalg.norm(a2[:3]), 1.0)
    assert_greater(a2[3], 0)
    assert_greater(np.pi, a2[3])

    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        pr.check_axis_angle, np.zeros(3))
    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        pr.check_axis_angle, np.zeros((3, 3)))


def test_check_compact_axis_angle():
    """Test input validation for compact axis-angle representation."""
    a_list = [0, 0, 0]
    a = pr.check_compact_axis_angle(a_list)
    assert_array_almost_equal(a_list, a)
    assert_equal(type(a), np.ndarray)
    assert_equal(a.dtype, np.float64)

    random_state = np.random.RandomState(0)
    a = pr.norm_vector(pr.random_vector(random_state, 3))
    a *= np.pi + random_state.randn() * 4.0 * np.pi
    a2 = pr.check_compact_axis_angle(a)
    pr.assert_compact_axis_angle_equal(a, a2)
    assert_greater(np.linalg.norm(a2), 0)
    assert_greater(np.pi, np.linalg.norm(a2))

    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        pr.check_compact_axis_angle, np.zeros(4))
    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        pr.check_compact_axis_angle, np.zeros((3, 3)))


def test_check_quaternion():
    """Test input validation for quaternion representation."""
    q_list = [1, 0, 0, 0]
    q = pr.check_quaternion(q_list)
    assert_array_almost_equal(q_list, q)
    assert_equal(type(q), np.ndarray)
    assert_equal(q.dtype, np.float64)

    random_state = np.random.RandomState(0)
    q = random_state.randn(4)
    q = pr.check_quaternion(q)
    assert_almost_equal(np.linalg.norm(q), 1.0)

    assert_raises_regexp(ValueError, "Expected quaternion with shape",
                         pr.check_quaternion, np.zeros(3))
    assert_raises_regexp(ValueError, "Expected quaternion with shape",
                         pr.check_quaternion, np.zeros((3, 3)))

    q = np.array([0.0, 1.2, 0.0, 0.0])
    q2 = pr.check_quaternion(q, unit=False)
    assert_array_almost_equal(q, q2)


def test_check_quaternions():
    """Test input validation for sequence of quaternions."""
    Q_list = [[1, 0, 0, 0]]
    Q = pr.check_quaternions(Q_list)
    assert_array_almost_equal(Q_list, Q)
    assert_equal(type(Q), np.ndarray)
    assert_equal(Q.dtype, np.float64)
    assert_equal(Q.ndim, 2)
    assert_array_equal(Q.shape, (1, 4))

    Q = np.array([
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
        [5, 0, 0, 0]
    ])
    Q = pr.check_quaternions(Q)
    for i in range(len(Q)):
        assert_almost_equal(np.linalg.norm(Q[i]), 1)

    assert_raises_regexp(ValueError, "Expected quaternion array with shape",
                         pr.check_quaternions, np.zeros(4))
    assert_raises_regexp(ValueError, "Expected quaternion array with shape",
                         pr.check_quaternions, np.zeros((3, 3)))

    Q = np.array([[0.0, 1.2, 0.0, 0.0]])
    Q2 = pr.check_quaternions(Q, unit=False)
    assert_array_almost_equal(Q, Q2)


def test_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    assert_raises_regexp(ValueError, "Basis must be in", pr.matrix_from_angle,
                         -1, 0)
    assert_raises_regexp(ValueError, "Basis must be in", pr.matrix_from_angle,
                         3, 0)

    R = pr.matrix_from_angle(0, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    R = pr.matrix_from_angle(0, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    R = pr.matrix_from_angle(1, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    R = pr.matrix_from_angle(1, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))

    R = pr.matrix_from_angle(2, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    R = pr.matrix_from_angle(2, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


def test_active_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    assert_raises_regexp(ValueError, "Basis must be in",
                         pr.active_matrix_from_angle, -1, 0)
    assert_raises_regexp(ValueError, "Basis must be in",
                         pr.active_matrix_from_angle, 3, 0)

    random_state = np.random.RandomState(21)
    for i in range(20):
        basis = random_state.randint(0, 3)
        angle = 2.0 * np.pi * random_state.rand() - np.pi
        R_passive = pr.passive_matrix_from_angle(basis, angle)
        R_active = pr.active_matrix_from_angle(basis, angle)
        assert_array_almost_equal(R_active, R_passive.T)


def test_conversions_matrix_euler_xyz():
    """Test conversions between rotation matrix and xyz Euler angles."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        R = pr.matrix_from_axis_angle(a)
        pr.assert_rotation_matrix(R)

        e_xyz = pr.euler_xyz_from_matrix(R)
        R2 = pr.matrix_from_euler_xyz(e_xyz)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)

        e_xyz2 = pr.euler_xyz_from_matrix(R2)
        pr.assert_euler_xyz_equal(e_xyz, e_xyz2)

    # Gimbal lock
    for _ in range(5):
        e_xyz = random_state.rand(3)
        e_xyz[1] = np.pi / 2.0
        R = pr.matrix_from_euler_xyz(e_xyz)
        e_xyz2 = pr.euler_xyz_from_matrix(R)
        pr.assert_euler_xyz_equal(e_xyz, e_xyz2)

        e_xyz[1] = -np.pi / 2.0
        R = pr.matrix_from_euler_xyz(e_xyz)
        e_xyz2 = pr.euler_xyz_from_matrix(R)
        pr.assert_euler_xyz_equal(e_xyz, e_xyz2)


def test_conversions_matrix_euler_zyx():
    """Test conversions between rotation matrix and zyx Euler angles."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        R = pr.matrix_from_axis_angle(a)
        pr.assert_rotation_matrix(R)

        e_zyx = pr.euler_zyx_from_matrix(R)
        R2 = pr.matrix_from_euler_zyx(e_zyx)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)

        e_zyx2 = pr.euler_zyx_from_matrix(R2)
        pr.assert_euler_zyx_equal(e_zyx, e_zyx2)

    # Gimbal lock
    for _ in range(5):
        e_zyx = random_state.rand(3)
        e_zyx[1] = np.pi / 2.0
        R = pr.matrix_from_euler_zyx(e_zyx)
        e_zyx2 = pr.euler_zyx_from_matrix(R)
        pr.assert_euler_zyx_equal(e_zyx, e_zyx2)

        e_zyx[1] = -np.pi / 2.0
        R = pr.matrix_from_euler_zyx(e_zyx)
        e_zyx2 = pr.euler_zyx_from_matrix(R)
        pr.assert_euler_zyx_equal(e_zyx, e_zyx2)


def test_active_matrix_from_intrinsic_euler_zxz():
    """Test conversion from intrinsic zxz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz(
            [0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    )


def test_active_matrix_from_extrinsic_euler_zxz():
    """Test conversion from extrinsic zxz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz(
            [0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zxz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    )


def test_active_matrix_from_intrinsic_euler_zyz():
    """Test conversion from intrinsic zyz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz(
            [0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_intrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    )


def test_active_matrix_from_extrinsic_euler_zyz():
    """Test conversion from roll, pitch, yaw."""
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw([0.5 * np.pi, 0, 0]),
        np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw(
            [0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw(
            [0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_roll_pitch_yaw(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    )


def test_active_matrix_from_intrinsic_zyx():
    """Test conversion from intrinsic zyx Euler angles."""
    random_state = np.random.RandomState(844)
    for _ in range(5):
        euler_zyx = ((random_state.rand(3) - 0.5) *
                     np.array([np.pi, 0.5 * np.pi, np.pi]))
        s = np.sin(euler_zyx)
        c = np.cos(euler_zyx)
        R_from_formula = np.array([
            [c[0] * c[1], c[0] * s[1] * s[2] - s[0] * c[2],
             c[0] * s[1] * c[2] + s[0] * s[2]],
            [s[0] * c[1], s[0] * s[1] * s[2] + c[0] * c[2],
             s[0] * s[1] * c[2] - c[0] * s[2]],
            [-s[1], c[1] * s[2], c[1] * c[2]]
        ])  # See Lynch, Park: Modern Robotics, page 576

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
        assert_almost_equal(euler_zyx2[1], 0.5 * np.pi)
        assert_almost_equal(
            euler_zyx[0] - euler_zyx[2], euler_zyx2[0] - euler_zyx2[2])

        # Gimbal lock 2, infinite solutions with constraint
        # alpha + gamma = constant
        euler_zyx[1] = -0.5 * np.pi
        R = pr.active_matrix_from_intrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.intrinsic_euler_zyx_from_active_matrix(R)
        assert_almost_equal(euler_zyx2[1], -0.5 * np.pi)
        assert_almost_equal(
            euler_zyx[0] + euler_zyx[2], euler_zyx2[0] + euler_zyx2[2])


def test_active_matrix_from_extrinsic_zyx():
    """Test conversion from extrinsic zyx Euler angles."""
    random_state = np.random.RandomState(844)
    for _ in range(5):
        euler_zyx = ((random_state.rand(3) - 0.5)
                     * np.array([np.pi, 0.5 * np.pi, np.pi]))

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
        assert_almost_equal(euler_zyx2[1], 0.5 * np.pi)
        assert_almost_equal(
            euler_zyx[0] + euler_zyx[2], euler_zyx2[0] + euler_zyx2[2])
        R2 = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx2)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 2, infinite solutions with constraint
        # alpha - gamma = constant
        euler_zyx[1] = -0.5 * np.pi
        R = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx)
        euler_zyx2 = pr.extrinsic_euler_zyx_from_active_matrix(R)
        assert_almost_equal(euler_zyx2[1], -0.5 * np.pi)
        assert_almost_equal(
            euler_zyx[0] - euler_zyx[2], euler_zyx2[0] - euler_zyx2[2])
        R2 = pr.active_matrix_from_extrinsic_euler_zyx(euler_zyx2)
        assert_array_almost_equal(R, R2)


def _test_conversion_matrix_euler(
        matrix_from_euler, euler_from_matrix, proper_euler):
    """Test conversions between Euler angles and rotation matrix."""
    random_state = np.random.RandomState(844)
    for _ in range(5):
        euler = ((random_state.rand(3) - 0.5)
                 * np.array([np.pi, 0.5 * np.pi, np.pi]))
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
        assert_almost_equal(euler[1], euler2[1])
        R2 = matrix_from_euler(euler2)
        assert_array_almost_equal(R, R2)

        # Gimbal lock 2
        if proper_euler:
            euler[1] = 0.0
        else:
            euler[1] = -0.5 * np.pi
        R = matrix_from_euler(euler)
        euler2 = euler_from_matrix(R)
        assert_almost_equal(euler[1], euler2[1])
        R2 = matrix_from_euler(euler2)
        assert_array_almost_equal(R, R2)


def test_all_euler_matrix_conversions():
    """Test all conversion between Euler angles and matrices."""
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xzx,
        pr.intrinsic_euler_xzx_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xzx,
        pr.extrinsic_euler_xzx_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xyx,
        pr.intrinsic_euler_xyx_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xyx,
        pr.extrinsic_euler_xyx_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yxy,
        pr.intrinsic_euler_yxy_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yxy,
        pr.extrinsic_euler_yxy_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yzy,
        pr.intrinsic_euler_yzy_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yzy,
        pr.extrinsic_euler_yzy_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zyz,
        pr.intrinsic_euler_zyz_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zyz,
        pr.extrinsic_euler_zyz_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zxz,
        pr.intrinsic_euler_zxz_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zxz,
        pr.extrinsic_euler_zxz_from_active_matrix,
        proper_euler=True)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xzy,
        pr.intrinsic_euler_xzy_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xzy,
        pr.extrinsic_euler_xzy_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_xyz,
        pr.intrinsic_euler_xyz_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_xyz,
        pr.extrinsic_euler_xyz_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yxz,
        pr.intrinsic_euler_yxz_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yxz,
        pr.extrinsic_euler_yxz_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_yzx,
        pr.intrinsic_euler_yzx_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_yzx,
        pr.extrinsic_euler_yzx_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zyx,
        pr.intrinsic_euler_zyx_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zyx,
        pr.extrinsic_euler_zyx_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_intrinsic_euler_zxy,
        pr.intrinsic_euler_zxy_from_active_matrix,
        proper_euler=False)
    _test_conversion_matrix_euler(
        pr.active_matrix_from_extrinsic_euler_zxy,
        pr.extrinsic_euler_zxy_from_active_matrix,
        proper_euler=False)


def test_active_matrix_from_extrinsic_roll_pitch_yaw():
    """Test conversion from extrinsic zyz Euler angles."""
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz(
            [0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        pr.active_matrix_from_extrinsic_euler_zyz(
            [0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    )


def test_conversions_matrix_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    R = np.eye(3)
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, -np.pi, 0.0]))
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, 0.0, -np.pi]))
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([0, 1, 0, np.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([0.0, -np.pi, -np.pi]))
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, np.pi]))

    a = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, np.pi])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a2, a)

    random_state = np.random.RandomState(0)
    for _ in range(50):
        a = pr.random_axis_angle(random_state)
        R = pr.matrix_from_axis_angle(a)
        pr.assert_rotation_matrix(R)

        a2 = pr.axis_angle_from_matrix(R)
        pr.assert_axis_angle_equal(a, a2)

        R2 = pr.matrix_from_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_compare_axis_angle_from_matrix_to_lynch_park():
    """Compare log(R) to the version of Lynch, Park: Modern Robotics."""
    R = np.eye(3)
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, [0, 0, 0, 0])

    R = pr.matrix_from_angle(2, np.pi)
    assert_almost_equal(np.trace(R), -1)
    a = pr.axis_angle_from_matrix(R)
    axis = (1.0 / np.sqrt(2.0 * (1 + R[2, 2]))
            * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]]))
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    R = pr.matrix_from_angle(1, np.pi)
    assert_almost_equal(np.trace(R), -1)
    a = pr.axis_angle_from_matrix(R)
    axis = (1.0 / np.sqrt(2.0 * (1 + R[1, 1]))
            * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]]))
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    R = pr.matrix_from_angle(0, np.pi)
    assert_almost_equal(np.trace(R), -1)
    a = pr.axis_angle_from_matrix(R)
    axis = (1.0 / np.sqrt(2.0 * (1 + R[0, 0]))
            * np.array([1 + R[0, 0], R[1, 0], R[2, 0]]))
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    # normal case is omitted here


def test_conversions_matrix_compact_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    R = np.eye(3)
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.zeros(3))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, -np.pi, 0.0]))
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.array([0, 0, np.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, 0.0, -np.pi]))
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.array([0, np.pi, 0]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([0.0, -np.pi, -np.pi]))
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.array([np.pi, 0, 0]))

    a = np.array([np.sqrt(0.5) * np.pi, np.sqrt(0.5) * np.pi, 0.0])
    R = pr.matrix_from_compact_axis_angle(a)
    a2 = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a2, a)

    random_state = np.random.RandomState(0)
    for _ in range(50):
        a = pr.random_compact_axis_angle(random_state)
        R = pr.matrix_from_compact_axis_angle(a)
        pr.assert_rotation_matrix(R)

        a2 = pr.compact_axis_angle_from_matrix(R)
        pr.assert_compact_axis_angle_equal(a, a2)

        R2 = pr.matrix_from_compact_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_active_rotation_is_default():
    """Test that rotations are active by default."""
    Rx = pr.active_matrix_from_angle(0, 0.5 * np.pi)
    ax = np.array([1, 0, 0, 0.5 * np.pi])
    qx = pr.quaternion_from_axis_angle(ax)
    assert_array_almost_equal(Rx, pr.matrix_from_axis_angle(ax))
    assert_array_almost_equal(Rx, pr.matrix_from_quaternion(qx))
    Ry = pr.active_matrix_from_angle(1, 0.5 * np.pi)
    ay = np.array([0, 1, 0, 0.5 * np.pi])
    qy = pr.quaternion_from_axis_angle(ay)
    assert_array_almost_equal(Ry, pr.matrix_from_axis_angle(ay))
    assert_array_almost_equal(Ry, pr.matrix_from_quaternion(qy))
    Rz = pr.active_matrix_from_angle(2, 0.5 * np.pi)
    az = np.array([0, 0, 1, 0.5 * np.pi])
    qz = pr.quaternion_from_axis_angle(az)
    assert_array_almost_equal(Rz, pr.matrix_from_axis_angle(az))
    assert_array_almost_equal(Rz, pr.matrix_from_quaternion(qz))


def test_issue43():
    """Test axis_angle_from_matrix() with angles close to 0 and pi."""
    a = np.array([-1., 1., 1., np.pi - 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., np.pi + 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., -5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)


def test_issue43_numerical_precision():
    """Test numerical precision of angles close to 0 and pi."""
    a = np.array([1., 1., 1., np.pi - 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert_less(axis_dist, 1e-10)
    assert_less(abs(a[3] - a2[3]), 1e-8)

    a = np.array([1., 1., 1., 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert_less(axis_dist, 1e-10)
    assert_less(abs(a[3] - a2[3]), 1e-8)


def test_conversions_matrix_axis_angle_continuous():
    """Test continuous conversions between rotation matrix and axis-angle."""
    for angle in np.arange(3.1, 3.2, 0.01):
        a = np.array([1.0, 0.0, 0.0, angle])
        R = pr.matrix_from_axis_angle(a)
        pr.assert_rotation_matrix(R)

        a2 = pr.axis_angle_from_matrix(R)
        pr.assert_axis_angle_equal(a, a2)

        R2 = pr.matrix_from_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_conversions_matrix_quaternion():
    """Test conversions between rotation matrix and quaternion."""
    R = np.eye(3)
    a = pr.axis_angle_from_matrix(R)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))

    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        R = pr.matrix_from_quaternion(q)
        pr.assert_rotation_matrix(R)

        q2 = pr.quaternion_from_matrix(R)
        pr.assert_quaternion_equal(q, q2)

        R2 = pr.matrix_from_quaternion(q2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_matrix_from_quaternion_hamilton():
    """Test if the conversion from quaternion to matrix is Hamiltonian."""
    q = np.sqrt(0.5) * np.array([1, 0, 0, 1])
    R = pr.matrix_from_quaternion(q)
    assert_array_almost_equal(
        np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]),
        R
    )


def test_quaternion_from_matrix_180():
    """Test for bug in conversion from 180 degree rotations."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pr.quaternion_from_matrix(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pr.quaternion_from_matrix(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pr.quaternion_from_matrix(R)
    assert_array_almost_equal(q, q_from_R)

    R = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0]])
    assert_raises_regexp(
        ValueError, "Expected rotation matrix", pr.quaternion_from_matrix, R)

    R = np.array(
        [[-1.0, 0.0, 0.0],
         [0.0, 0.00000001, 1.0],
         [0.0, 1.0, -0.00000001]])
    q_from_R = pr.quaternion_from_matrix(R)


def test_quaternion_from_matrix_180_not_axis_aligned():
    """Test for bug in rotation by 180 degrees around arbitrary axes."""
    random_state = np.random.RandomState(0)
    for i in range(10):
        a = pr.random_axis_angle(random_state)
        a[3] = np.pi
        q = pr.quaternion_from_axis_angle(a)
        R = pr.matrix_from_axis_angle(a)
        q_from_R = pr.quaternion_from_matrix(R)
        pr.assert_quaternion_equal(q, q_from_R)


def test_conversions_axis_angle_quaternion():
    """Test conversions between axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = pr.axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))
    q2 = pr.quaternion_from_axis_angle(a)
    assert_array_almost_equal(q2, q)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        q = pr.quaternion_from_axis_angle(a)

        a2 = pr.axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = pr.quaternion_from_axis_angle(a2)
        pr.assert_quaternion_equal(q, q2)


def test_axis_angle_from_two_direction_vectors():
    """Test calculation of axis-angle from two direction vectors."""
    d1 = np.array([1.0, 0.0, 0.0])
    a = pr.axis_angle_from_two_directions(d1, d1)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    a = pr.axis_angle_from_two_directions(d1, np.zeros(3))
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    a = pr.axis_angle_from_two_directions(np.zeros(3), d1)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    d2 = np.array([0.0, 1.0, 0.0])
    a = pr.axis_angle_from_two_directions(d1, d2)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, 0.5 * np.pi]))

    d3 = np.array([-1.0, 0.0, 0.0])
    a = pr.axis_angle_from_two_directions(d1, d3)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    d4 = np.array([0.0, -1.0, 0.0])
    a = pr.axis_angle_from_two_directions(d2, d4)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    a = pr.axis_angle_from_two_directions(d3, d4)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, 0.5 * np.pi]))

    random_state = np.random.RandomState(323)
    for i in range(5):
        R = pr.matrix_from_axis_angle(pr.random_axis_angle(random_state))
        v1 = pr.random_vector(random_state, 3)
        v2 = R.dot(v1)
        a = pr.axis_angle_from_two_directions(v1, v2)
        assert_almost_equal(pr.angle_between_vectors(v1, a[:3]), 0.5 * np.pi)
        assert_almost_equal(pr.angle_between_vectors(v2, a[:3]), 0.5 * np.pi)
        assert_array_almost_equal(v2, pr.matrix_from_axis_angle(a).dot(v1))


def test_axis_angle_from_compact_axis_angle():
    """Test conversion from compact axis-angle representation."""
    ca = [0.0, 0.0, 0.0]
    a = pr.axis_angle_from_compact_axis_angle(ca)
    assert_array_almost_equal(a, np.array([1.0, 0.0, 0.0, 0.0]))

    random_state = np.random.RandomState(1)
    for _ in range(5):
        ca = pr.random_compact_axis_angle(random_state)
        a = pr.axis_angle_from_compact_axis_angle(ca)
        assert_almost_equal(np.linalg.norm(ca), a[3])
        assert_array_almost_equal(ca[:3] / np.linalg.norm(ca), a[:3])


def test_compact_axis_angle():
    """Test conversion to compact axis-angle representation."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    ca = pr.compact_axis_angle(a)
    assert_array_almost_equal(ca, np.zeros(3))

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        ca = pr.compact_axis_angle(a)
        assert_array_almost_equal(pr.norm_vector(ca), a[:3])
        assert_almost_equal(np.linalg.norm(ca), a[3])


def test_conversions_compact_axis_angle_quaternion():
    """Test conversions between compact axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = pr.compact_axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, np.array([0, 0, 0]))
    q2 = pr.quaternion_from_compact_axis_angle(a)
    assert_array_almost_equal(q2, q)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = pr.random_compact_axis_angle(random_state)
        q = pr.quaternion_from_compact_axis_angle(a)

        a2 = pr.compact_axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = pr.quaternion_from_compact_axis_angle(a2)
        pr.assert_quaternion_equal(q, q2)


def test_conversions_to_matrix():
    """Test conversions to rotation matrix."""
    R = np.eye(3)
    R2R = pr.matrix_from(R=R)
    assert_array_almost_equal(R2R, R)

    a = np.array([1, 0, 0, 0])
    a2R = pr.matrix_from(a=a)
    assert_array_almost_equal(a2R, R)

    q = np.array([1, 0, 0, 0])
    q2R = pr.matrix_from(q=q)
    assert_array_almost_equal(q2R, R)

    e_xyz = np.array([0, 0, 0])
    e_xyz2R = pr.matrix_from(e_xyz=e_xyz)
    assert_array_almost_equal(e_xyz2R, R)

    e_zyx = np.array([0, 0, 0])
    e_zyx2R = pr.matrix_from(e_zyx=e_zyx)
    assert_array_almost_equal(e_zyx2R, R)

    assert_raises_regexp(ValueError, "no rotation", pr.matrix_from)


def test_interpolate_axis_angle():
    """Test interpolation between two axis-angle rotations with slerp."""
    n_steps = 10
    random_state = np.random.RandomState(1)
    a1 = pr.random_axis_angle(random_state)
    a2 = pr.random_axis_angle(random_state)

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
    random_state = np.random.RandomState(42)
    a = pr.random_axis_angle(random_state)
    traj = [pr.axis_angle_slerp(a, a, t) for t in np.linspace(0, 1, n_steps)]
    assert_equal(len(traj), n_steps)
    assert_array_almost_equal(traj[0], a)
    assert_array_almost_equal(traj[1], a)
    assert_array_almost_equal(traj[2], a)


def test_interpolate_almost_same_axis_angle():
    """Test interpolation between almost the same axis-angle rotation."""
    n_steps = 3
    random_state = np.random.RandomState(42)
    a1 = pr.random_axis_angle(random_state)
    a2 = np.copy(a1)
    a2[-1] += np.finfo("float").eps
    traj = [pr.axis_angle_slerp(a1, a2, t) for t in np.linspace(0, 1, n_steps)]
    assert_equal(len(traj), n_steps)
    assert_array_almost_equal(traj[0], a1)
    assert_array_almost_equal(traj[1], a1)
    assert_array_almost_equal(traj[2], a2)


def test_interpolate_quaternion():
    """Test interpolation between two quaternions with slerp."""
    n_steps = 10
    random_state = np.random.RandomState(0)
    a1 = pr.random_axis_angle(random_state)
    a2 = pr.random_axis_angle(random_state)
    q1 = pr.quaternion_from_axis_angle(a1)
    q2 = pr.quaternion_from_axis_angle(a2)

    traj_q = [pr.quaternion_slerp(q1, q2, t)
              for t in np.linspace(0, 1, n_steps)]
    traj_R = [pr.matrix_from_quaternion(q) for q in traj_q]
    R_diff = np.diff(traj_R, axis=0)
    R_diff_norms = [np.linalg.norm(Rd) for Rd in R_diff]
    assert_array_almost_equal(R_diff_norms,
                              R_diff_norms[0] * np.ones(n_steps - 1))


def test_interpolate_same_quaternion():
    """Test interpolation between the same quaternion rotation.

    See issue #45.
    """
    n_steps = 3
    random_state = np.random.RandomState(42)
    a = pr.random_axis_angle(random_state)
    q = pr.quaternion_from_axis_angle(a)
    traj = [pr.quaternion_slerp(q, q, t) for t in np.linspace(0, 1, n_steps)]
    assert_equal(len(traj), n_steps)
    assert_array_almost_equal(traj[0], q)
    assert_array_almost_equal(traj[1], q)
    assert_array_almost_equal(traj[2], q)


def test_quaternion_conventions():
    """Test conversion of quaternion between wxyz and xyzw."""
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    q_xyzw = pr.quaternion_xyzw_from_wxyz(q_wxyz)
    assert_array_equal(q_xyzw, np.array([0.0, 0.0, 0.0, 1.0]))
    q_wxyz2 = pr.quaternion_wxyz_from_xyzw(q_xyzw)
    assert_array_equal(q_wxyz, q_wxyz2)

    random_state = np.random.RandomState(42)
    q_wxyz_random = pr.random_quaternion(random_state)
    q_xyzw_random = pr.quaternion_xyzw_from_wxyz(q_wxyz_random)
    assert_array_equal(q_xyzw_random[:3], q_wxyz_random[1:])
    assert_equal(q_xyzw_random[3], q_wxyz_random[0])
    q_wxyz_random2 = pr.quaternion_wxyz_from_xyzw(q_xyzw_random)
    assert_array_equal(q_wxyz_random, q_wxyz_random2)


def test_concatenate_quaternions():
    """Test concatenation of two quaternions."""
    # Until ea9adc5, this combination of a list and a numpy array raised
    # a ValueError:
    q1 = [1, 0, 0, 0]
    q2 = np.array([0, 0, 0, 1])
    q12 = pr.concatenate_quaternions(q1, q2)
    assert_array_almost_equal(q12, np.array([0, 0, 0, 1]))

    random_state = np.random.RandomState(0)
    for _ in range(5):
        q1 = pr.quaternion_from_axis_angle(pr.random_axis_angle(random_state))
        q2 = pr.quaternion_from_axis_angle(pr.random_axis_angle(random_state))

        R1 = pr.matrix_from_quaternion(q1)
        R2 = pr.matrix_from_quaternion(q2)

        q12 = pr.concatenate_quaternions(q1, q2)
        R12 = np.dot(R1, R2)
        q12R = pr.quaternion_from_matrix(R12)

        pr.assert_quaternion_equal(q12, q12R)


def test_quaternion_hamilton():
    """Test if quaternion multiplication follows Hamilton's convention."""
    q_ij = pr.concatenate_quaternions(pr.q_i, pr.q_j)
    assert_array_equal(pr.q_k, q_ij)
    q_ijk = pr.concatenate_quaternions(q_ij, pr.q_k)
    assert_array_equal(-pr.q_id, q_ijk)


def test_quaternion_rotation():
    """Test quaternion rotation."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = pr.quaternion_from_axis_angle(pr.random_axis_angle(random_state))
        R = pr.matrix_from_quaternion(q)
        v = pr.random_vector(random_state)
        vR = np.dot(R, v)
        vq = pr.q_prod_vector(q, v)
        assert_array_almost_equal(vR, vq)


def test_quaternion_conjugate():
    """Test quaternion conjugate."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        v = pr.random_vector(random_state)
        vq = pr.q_prod_vector(q, v)
        vq2 = pr.concatenate_quaternions(pr.concatenate_quaternions(
            q, np.hstack(([0], v))), pr.q_conj(q))[1:]
        assert_array_almost_equal(vq, vq2)


def test_quaternion_invert():
    """Test unit quaternion inversion with conjugate."""
    q = np.array([0.58183503, -0.75119889, -0.24622332, 0.19116072])
    q_inv = pr.q_conj(q)
    q_q_inv = pr.concatenate_quaternions(q, q_inv)
    assert_array_almost_equal(pr.q_id, q_q_inv)


def test_quaternion_gradient_integration():
    """Test integration of quaternion gradients."""
    n_steps = 21
    dt = 0.1
    random_state = np.random.RandomState(3)
    for _ in range(5):
        q1 = pr.random_quaternion(random_state)
        q2 = pr.random_quaternion(random_state)
        Q = np.vstack([pr.quaternion_slerp(q1, q2, t)
                       for t in np.linspace(0, 1, n_steps)])
        angular_velocities = pr.quaternion_gradient(Q, dt)
        Q2 = pr.quaternion_integrate(angular_velocities, q1, dt)
        assert_array_almost_equal(Q, Q2)


def test_quaternion_rotation_consistent_with_multiplication():
    """Test if quaternion rotation and multiplication are Hamiltonian."""
    random_state = np.random.RandomState(1)
    for _ in range(5):
        v = pr.random_vector(random_state)
        q = pr.random_quaternion(random_state)
        v_im = np.hstack(((0.0,), v))
        qv_mult = pr.concatenate_quaternions(
            q, pr.concatenate_quaternions(v_im, pr.q_conj(q)))[1:]
        qv_rot = pr.q_prod_vector(q, v)
        assert_array_almost_equal(qv_mult, qv_rot)


def test_quaternion_dist():
    """Test angular metric of quaternions."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q1 = pr.quaternion_from_axis_angle(pr.random_axis_angle(random_state))
        q2 = pr.quaternion_from_axis_angle(pr.random_axis_angle(random_state))
        q1_to_q1 = pr.quaternion_dist(q1, q1)
        assert_almost_equal(q1_to_q1, 0.0)
        q2_to_q2 = pr.quaternion_dist(q2, q2)
        assert_almost_equal(q2_to_q2, 0.0)
        q1_to_q2 = pr.quaternion_dist(q1, q2)
        q2_to_q1 = pr.quaternion_dist(q2, q1)
        assert_almost_equal(q1_to_q2, q2_to_q1)
        assert_greater(2.0 * np.pi, q1_to_q2)


def test_quaternion_dist_for_identical_rotations():
    """Test angular metric of quaternions q and -q."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q = pr.quaternion_from_axis_angle(pr.random_axis_angle(random_state))
        assert_array_almost_equal(pr.matrix_from_quaternion(q),
                                  pr.matrix_from_quaternion(-q))
        assert_equal(pr.quaternion_dist(q, -q), 0.0)


def test_quaternion_dist_for_almost_identical_rotations():
    """Test angular metric of quaternions q and ca. -q."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        q1 = pr.quaternion_from_axis_angle(a)
        r = 1e-4 * random_state.randn(4)
        q2 = -pr.quaternion_from_axis_angle(a + r)
        assert_almost_equal(pr.quaternion_dist(q1, q2), 0.0, places=3)


def test_quaternion_diff():
    """Test difference of quaternions."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q1 = pr.random_quaternion(random_state)
        q2 = pr.random_quaternion(random_state)
        a_diff = pr.quaternion_diff(q1, q2)          # q1 - q2
        q_diff = pr.quaternion_from_axis_angle(a_diff)
        q3 = pr.concatenate_quaternions(q_diff, q2)  # q1 - q2 + q2
        pr.assert_quaternion_equal(q1, q3)


def test_id_rot():
    """Test equivalence of constants that represent no rotation."""
    assert_array_almost_equal(pr.R_id, pr.matrix_from_axis_angle(pr.a_id))
    assert_array_almost_equal(pr.R_id, pr.matrix_from_quaternion(pr.q_id))
    assert_array_almost_equal(pr.R_id, pr.matrix_from_euler_xyz(pr.e_xyz_id))
    assert_array_almost_equal(pr.R_id, pr.matrix_from_euler_zyx(pr.e_zyx_id))


def test_check_matrix_threshold():
    """Test matrix threshold.

    See issue #54.
    """
    R = np.array([
        [-9.15361835e-01, 4.01808328e-01, 2.57475872e-02],
        [5.15480570e-02, 1.80374088e-01, -9.82246499e-01],
        [-3.99318925e-01, -8.97783496e-01, -1.85819250e-01]])
    pr.assert_rotation_matrix(R)
    pr.check_matrix(R)


def test_asssert_rotation_matrix_behaves_like_check_matrix():
    """Test of both checks for rotation matrix validity behave similar."""
    random_state = np.random.RandomState(2345)
    for _ in range(5):
        a = pr.random_axis_angle(random_state)
        R = pr.matrix_from_axis_angle(a)
        original_value = R[2, 2]
        for error in [0, 1e-8, 1e-7, 1e-5, 1e-4, 1]:
            R[2, 2] = original_value + error
            try:
                pr.assert_rotation_matrix(R)
                pr.check_matrix(R)
            except AssertionError:
                assert_raises_regexp(
                    ValueError, "Expected rotation matrix", pr.check_matrix, R)


def test_deactivate_rotation_matrix_precision_error():
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    assert_raises_regexp(
        ValueError, "Expected rotation matrix", pr.check_matrix, R)
    with warnings.catch_warnings(record=True) as w:
        pr.check_matrix(R, strict_check=False)
        assert_equal(len(w), 2)


def test_norm_rotation_matrix():
    R = pr.norm_matrix(np.eye(3))
    assert_array_equal(R, np.eye(3))

    R[1, 0] += np.finfo(float).eps
    R = pr.norm_matrix(R)
    assert_array_equal(R, np.eye(3))
    assert_equal(np.linalg.det(R), 1.0)

    R = np.eye(3)
    R[1, 1] += 0.3
    R_norm = pr.norm_matrix(R)
    assert_almost_equal(np.linalg.det(R_norm), 1.0)
    assert_array_almost_equal(np.eye(3), R_norm)


def test_matrix_from_two_vectors():
    assert_raises_regexp(
        ValueError, "a must not be the zero vector",
        pr.matrix_from_two_vectors, np.zeros(3), np.zeros(3))
    assert_raises_regexp(
        ValueError, "b must not be the zero vector",
        pr.matrix_from_two_vectors, np.ones(3), np.zeros(3))
    assert_raises_regexp(
        ValueError, "a and b must not be parallel",
        pr.matrix_from_two_vectors, np.ones(3), np.ones(3))

    R = pr.matrix_from_two_vectors(pr.unitx, pr.unity)
    assert_array_almost_equal(R, np.eye(3))

    random_state = np.random.RandomState(28)
    for _ in range(5):
        a = pr.random_vector(random_state, 3)
        b = pr.random_vector(random_state, 3)
        R = pr.matrix_from_two_vectors(a, b)
        pr.assert_rotation_matrix(R)
        assert_array_almost_equal(pr.norm_vector(a), R[:, 0])
        assert_almost_equal(pr.angle_between_vectors(b, R[:, 2]), 0.5 * np.pi)


def test_axis_angle_from_matrix_cos_angle_greater_1():
    R = np.array([
        [1.0000000000000004, -1.4402617650886727e-08, 2.3816502339526408e-08],
        [1.4402617501592725e-08, 1.0000000000000004, 1.2457848566326355e-08],
        [-2.3816502529500374e-08, -1.2457848247850049e-08, 0.9999999999999999]])
    a = pr.axis_angle_from_matrix(R)
    assert_false(any(np.isnan(a)))


def test_axis_angle_from_matrix_without_check():
    R = -np.eye(3)
    with warnings.catch_warnings(record=True) as w:
        a = pr.axis_angle_from_matrix(R, check=False)
    assert_equal(len(w), 1)
    assert_true(all(np.isnan(a[:3])))
