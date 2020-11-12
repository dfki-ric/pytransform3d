import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import (assert_almost_equal, assert_equal, assert_true,
                        assert_greater, assert_raises_regexp, assert_less)
from pytransform3d.rotations import *


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


def test_norm_angle():
    """Test normalization of angle."""
    random_state = np.random.RandomState(0)
    a_norm = random_state.uniform(-np.pi, np.pi, size=(100,))
    for b in np.linspace(-10.0 * np.pi, 10.0 * np.pi, 11):
        a = a_norm + b
        assert_array_almost_equal(norm_angle(a), a_norm)

    assert_almost_equal(norm_angle(-np.pi), np.pi)
    assert_almost_equal(norm_angle(np.pi), np.pi)


def test_norm_axis_angle():
    """Test normalization of angle-axis representation."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    n = norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    n = norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    n = norm_axis_angle(a)
    assert_array_almost_equal(a, n)

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


def test_norm_compact_axis_angle():
    """Test normalization of compact angle-axis representation."""
    a = np.array([np.pi, 0.0, 0.0])
    n = norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, np.pi, 0.0])
    n = norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, np.pi])
    n = norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_compact_axis_angle(random_state)
        axis = a / np.linalg.norm(a)
        angle = random_state.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a = axis * angle
        n = norm_compact_axis_angle(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a = axis * (angle + angle_offset)
            n2 = norm_compact_axis_angle(a)
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


def test_angle_between_close_vectors():
    """Test angle between close vectors.

    See issue #47.
    """
    a = np.array([0.9689124217106448 , 0.24740395925452294 , 0.0 , 0.0])
    b = np.array([0.9689124217106448 , 0.247403959254523 , 0.0 , 0.0])
    angle = angle_between_vectors(a, b)
    assert_almost_equal(angle, 0.0)


def test_angle_to_zero_vector_is_nan():
    """Test angle to zero vector."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    with warnings.catch_warnings(record=True) as w:
        angle = angle_between_vectors(a, b)
        assert_equal(len(w), 1)
    assert_true(np.isnan(angle))


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


def test_check_matrix():
    """Test input validation for rotation matrix."""
    R_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    R = check_matrix(R_list)
    assert_equal(type(R), np.ndarray)
    assert_equal(R.dtype, np.float)

    R_int_array = np.eye(3, dtype=np.int)
    R = check_matrix(R_int_array)
    assert_equal(type(R), np.ndarray)
    assert_equal(R.dtype, np.float)

    R_array = np.eye(3)
    R = check_matrix(R_array)
    assert_array_equal(R_array, R)

    R = np.eye(4)
    assert_raises_regexp(
        ValueError, "Expected rotation matrix with shape",
        check_matrix, R)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0.1, 1]])
    assert_raises_regexp(
        ValueError, "inversion by transposition", check_matrix, R)

    R = np.array([[1, 0, 1e-16], [0, 1, 0], [0, 0, 1]])
    R2 = check_matrix(R)
    assert_array_equal(R, R2)

    R = -np.eye(3)
    assert_raises_regexp(ValueError, "determinant", check_matrix, R)


def test_check_axis_angle():
    """Test input validation for axis-angle representation."""
    a_list = [1, 0, 0, 0]
    a = check_axis_angle(a_list)
    assert_array_almost_equal(a_list, a)
    assert_equal(type(a), np.ndarray)
    assert_equal(a.dtype, np.float)

    random_state = np.random.RandomState(0)
    a = np.empty(4)
    a[:3] = random_vector(random_state, 3)
    a[3] = random_state.randn() * 4.0 * np.pi
    a2 = check_axis_angle(a)
    assert_axis_angle_equal(a, a2)
    assert_almost_equal(np.linalg.norm(a2[:3]), 1.0)
    assert_greater(a2[3], 0)
    assert_greater(np.pi, a2[3])

    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        check_axis_angle, np.zeros(3))
    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        check_axis_angle, np.zeros((3, 3)))


def test_check_compact_axis_angle():
    """Test input validation for compact axis-angle representation."""
    a_list = [0, 0, 0]
    a = check_compact_axis_angle(a_list)
    assert_array_almost_equal(a_list, a)
    assert_equal(type(a), np.ndarray)
    assert_equal(a.dtype, np.float)

    random_state = np.random.RandomState(0)
    a = norm_vector(random_vector(random_state, 3))
    a *= np.pi + random_state.randn() * 4.0 * np.pi
    a2 = check_compact_axis_angle(a)
    assert_compact_axis_angle_equal(a, a2)
    assert_greater(np.linalg.norm(a2), 0)
    assert_greater(np.pi, np.linalg.norm(a2))

    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        check_compact_axis_angle, np.zeros(4))
    assert_raises_regexp(
        ValueError, "Expected axis and angle in array with shape",
        check_compact_axis_angle, np.zeros((3, 3)))


def test_check_quaternion():
    """Test input validation for quaternion representation."""
    q_list = [1, 0, 0, 0]
    q = check_quaternion(q_list)
    assert_array_almost_equal(q_list, q)
    assert_equal(type(q), np.ndarray)
    assert_equal(q.dtype, np.float)

    random_state = np.random.RandomState(0)
    q = random_state.randn(4)
    q = check_quaternion(q)
    assert_almost_equal(np.linalg.norm(q), 1.0)

    assert_raises_regexp(ValueError, "Expected quaternion with shape",
                         check_quaternion, np.zeros(3))
    assert_raises_regexp(ValueError, "Expected quaternion with shape",
                         check_quaternion, np.zeros((3, 3)))

    q = np.array([0.0, 1.2, 0.0, 0.0])
    q2 = check_quaternion(q, unit=False)
    assert_array_almost_equal(q, q2)


def test_check_quaternions():
    """Test input validation for sequence of quaternions."""
    Q_list = [[1, 0, 0, 0]]
    Q = check_quaternions(Q_list)
    assert_array_almost_equal(Q_list, Q)
    assert_equal(type(Q), np.ndarray)
    assert_equal(Q.dtype, np.float)
    assert_equal(Q.ndim, 2)
    assert_array_equal(Q.shape, (1, 4))

    Q = np.array([
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [4, 0, 0, 0],
        [5, 0, 0, 0]
    ])
    Q = check_quaternions(Q)
    for i in range(len(Q)):
        assert_almost_equal(np.linalg.norm(Q[i]), 1)

    assert_raises_regexp(ValueError, "Expected quaternion array with shape",
                         check_quaternions, np.zeros(4))
    assert_raises_regexp(ValueError, "Expected quaternion array with shape",
                         check_quaternions, np.zeros((3, 3)))

    Q = np.array([[0.0, 1.2, 0.0, 0.0]])
    Q2 = check_quaternions(Q, unit=False)
    assert_array_almost_equal(Q, Q2)


def test_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    assert_raises_regexp(ValueError, "Basis must be in", matrix_from_angle,
                         -1, 0)
    assert_raises_regexp(ValueError, "Basis must be in", matrix_from_angle,
                         3, 0)

    R = matrix_from_angle(0, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    R = matrix_from_angle(0, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    R = matrix_from_angle(1, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    R = matrix_from_angle(1, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))

    R = matrix_from_angle(2, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    R = matrix_from_angle(2, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


def test_active_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    assert_raises_regexp(ValueError, "Basis must be in",
                         active_matrix_from_angle, -1, 0)
    assert_raises_regexp(ValueError, "Basis must be in",
                         active_matrix_from_angle, 3, 0)

    random_state = np.random.RandomState(21)
    for i in range(20):
        basis = random_state.randint(0, 3)
        angle = 2.0 * np.pi * random_state.rand() - np.pi
        R_passive = passive_matrix_from_angle(basis, angle)
        R_active = active_matrix_from_angle(basis, angle)
        assert_array_almost_equal(R_active, R_passive.T)


def test_conversions_matrix_euler_xyz():
    """Test conversions between rotation matrix and xyz Euler angles."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        e_xyz = euler_xyz_from_matrix(R)
        R2 = matrix_from_euler_xyz(e_xyz)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)

        e_xyz2 = euler_xyz_from_matrix(R2)
        assert_euler_xyz_equal(e_xyz, e_xyz2)

    # Gimbal lock
    for _ in range(5):
        e_xyz = random_state.rand(3)
        e_xyz[1] = np.pi / 2.0
        R = matrix_from_euler_xyz(e_xyz)
        e_xyz2 = euler_xyz_from_matrix(R)
        assert_euler_xyz_equal(e_xyz, e_xyz2)

        e_xyz[1] = -np.pi / 2.0
        R = matrix_from_euler_xyz(e_xyz)
        e_xyz2 = euler_xyz_from_matrix(R)
        assert_euler_xyz_equal(e_xyz, e_xyz2)


def test_conversions_matrix_euler_zyx():
    """Test conversions between rotation matrix and zyx Euler angles."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        e_zyx = euler_zyx_from_matrix(R)
        R2 = matrix_from_euler_zyx(e_zyx)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)

        e_zyx2 = euler_zyx_from_matrix(R2)
        assert_euler_zyx_equal(e_zyx, e_zyx2)

    # Gimbal lock
    for _ in range(5):
        e_zyx = random_state.rand(3)
        e_zyx[1] = np.pi / 2.0
        R = matrix_from_euler_zyx(e_zyx)
        e_zyx2 = euler_zyx_from_matrix(R)
        assert_euler_zyx_equal(e_zyx, e_zyx2)

        e_zyx[1] = -np.pi / 2.0
        R = matrix_from_euler_zyx(e_zyx)
        e_zyx2 = euler_zyx_from_matrix(R)
        assert_euler_zyx_equal(e_zyx, e_zyx2)


def test_active_matrix_from_intrinsic_euler_zxz():
    """Test conversion from intrinsic zxz Euler angles."""
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zxz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zxz([0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zxz([0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zxz([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    )


def test_active_matrix_from_extrinsic_euler_zxz():
    """Test conversion from extrinsic zxz Euler angles."""
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zxz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zxz([0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zxz([0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zxz([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0]
        ])
    )


def test_active_matrix_from_intrinsic_euler_zyz():
    """Test conversion from intrinsic zyz Euler angles."""
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_intrinsic_euler_zyz([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    )


def test_active_matrix_from_extrinsic_euler_zyz():
    """Test conversion from extrinsic zyz Euler angles."""
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zyz([0.5 * np.pi, 0, 0]),
        np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zyz([0.5 * np.pi, 0, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zyz([0.5 * np.pi, 0.5 * np.pi, 0]),
        np.array([
            [0, -1, 0],
            [0, 0, 1],
            [-1, 0, 0]
        ])
    )
    assert_array_almost_equal(
        active_matrix_from_extrinsic_euler_zyz([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]),
        np.array([
            [-1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])
    )


def test_conversions_matrix_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    R = np.eye(3)
    a = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    R = matrix_from_euler_xyz(np.array([np.pi, np.pi, 0.0]))
    a = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    R = matrix_from_euler_xyz(np.array([np.pi, 0.0, np.pi]))
    a = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, np.array([0, 1, 0, np.pi]))

    R = matrix_from_euler_xyz(np.array([0.0, np.pi, np.pi]))
    a = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, np.array([1, 0, 0, np.pi]))

    a = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, np.pi])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a2, a)

    random_state = np.random.RandomState(0)
    for _ in range(50):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        a2 = axis_angle_from_matrix(R)
        assert_axis_angle_equal(a, a2)

        R2 = matrix_from_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)


def test_conversions_matrix_compact_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    R = np.eye(3)
    a = compact_axis_angle_from_matrix(R)
    assert_compact_axis_angle_equal(a, np.zeros(3))

    R = matrix_from_euler_xyz(np.array([np.pi, np.pi, 0.0]))
    a = compact_axis_angle_from_matrix(R)
    assert_compact_axis_angle_equal(a, np.array([0, 0, np.pi]))

    R = matrix_from_euler_xyz(np.array([np.pi, 0.0, np.pi]))
    a = compact_axis_angle_from_matrix(R)
    assert_compact_axis_angle_equal(a, np.array([0, np.pi, 0]))

    R = matrix_from_euler_xyz(np.array([0.0, np.pi, np.pi]))
    a = compact_axis_angle_from_matrix(R)
    assert_compact_axis_angle_equal(a, np.array([np.pi, 0, 0]))

    a = np.array([np.sqrt(0.5) * np.pi, np.sqrt(0.5) * np.pi, 0.0])
    R = matrix_from_compact_axis_angle(a)
    a2 = compact_axis_angle_from_matrix(R)
    assert_compact_axis_angle_equal(a2, a)

    random_state = np.random.RandomState(0)
    for _ in range(50):
        a = random_compact_axis_angle(random_state)
        R = matrix_from_compact_axis_angle(a)
        assert_rotation_matrix(R)

        a2 = compact_axis_angle_from_matrix(R)
        assert_compact_axis_angle_equal(a, a2)

        R2 = matrix_from_compact_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)


def test_issue43():
    """Test axis_angle_from_matrix() with angles close to 0 and pi."""
    a = np.array([-1., 1., 1., np.pi - 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., np.pi + 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, a2)

    a = np.array([-1., 1., 1., -5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    assert_axis_angle_equal(a, a2)


def test_issue43_numerical_precision():
    """Test numerical precision of angles close to 0 and pi."""
    a = np.array([1., 1., 1., np.pi - 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert_less(axis_dist, 1e-10)
    assert_less(abs(a[3] - a2[3]), 1e-8)

    a = np.array([1., 1., 1., 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = matrix_from_axis_angle(a)
    a2 = axis_angle_from_matrix(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert_less(axis_dist, 1e-10)
    assert_less(abs(a[3] - a2[3]), 1e-8)


def test_conversions_matrix_axis_angle_continuous():
    """Test continuous conversions between rotation matrix and axis-angle."""
    for angle in np.arange(3.1, 3.2, 0.01):
        a = np.array([1.0, 0.0, 0.0, angle])
        R = matrix_from_axis_angle(a)
        assert_rotation_matrix(R)

        a2 = axis_angle_from_matrix(R)
        assert_axis_angle_equal(a, a2)

        R2 = matrix_from_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        assert_rotation_matrix(R2)


def test_conversions_matrix_quaternion():
    """Test conversions between rotation matrix and quaternion."""
    R = np.eye(3)
    a = axis_angle_from_matrix(R)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))

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


def test_matrix_from_quaternion_hamilton():
    """Test if the conversion from quaternion to matrix is Hamiltonian."""
    q = np.sqrt(0.5) * np.array([1, 0, 0, 1])
    R = matrix_from_quaternion(q)
    assert_array_almost_equal(
        np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]),
        R
    )


def test_quaternion_from_matrix_180():
    """Test for bug in conversion from 180 degree rotations."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    q = quaternion_from_axis_angle(a)
    R = matrix_from_axis_angle(a)
    q_from_R = quaternion_from_matrix(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    q = quaternion_from_axis_angle(a)
    R = matrix_from_axis_angle(a)
    q_from_R = quaternion_from_matrix(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    q = quaternion_from_axis_angle(a)
    R = matrix_from_axis_angle(a)
    q_from_R = quaternion_from_matrix(R)
    assert_array_almost_equal(q, q_from_R)

    R = np.array(
        [[0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0]])
    assert_raises_regexp(
        ValueError, "Expected rotation matrix", quaternion_from_matrix, R)

    R = np.array(
        [[-1.0, 0.0, 0.0],
         [0.0, 0.00000001, 1.0],
         [0.0, 1.0, -0.00000001]])
    q_from_R = quaternion_from_matrix(R)


def test_quaternion_from_matrix_180_not_axis_aligned():
    """Test for bug in rotation by 180 degrees around arbitrary axes."""
    random_state = np.random.RandomState(0)
    for i in range(10):
        a = random_axis_angle(random_state)
        a[3] = np.pi
        q = quaternion_from_axis_angle(a)
        R = matrix_from_axis_angle(a)
        q_from_R = quaternion_from_matrix(R)
        assert_quaternion_equal(q, q_from_R)


def test_conversions_axis_angle_quaternion():
    """Test conversions between axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))
    q2 = quaternion_from_axis_angle(a)
    assert_array_almost_equal(q2, q)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        q = quaternion_from_axis_angle(a)

        a2 = axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = quaternion_from_axis_angle(a2)
        assert_quaternion_equal(q, q2)


def test_axis_angle_from_compact_axis_angle():
    """Test conversion from compact axis-angle representation."""
    ca = [0.0, 0.0, 0.0]
    a = axis_angle_from_compact_axis_angle(ca)
    assert_array_almost_equal(a, np.array([1.0, 0.0, 0.0, 0.0]))

    random_state = np.random.RandomState(1)
    for _ in range(5):
        ca = random_compact_axis_angle(random_state)
        a = axis_angle_from_compact_axis_angle(ca)
        assert_almost_equal(np.linalg.norm(ca), a[3])
        assert_array_almost_equal(ca[:3] / np.linalg.norm(ca), a[:3])


def test_compact_axis_angle():
    """Test conversion to compact axis-angle representation."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    ca = compact_axis_angle(a)
    assert_array_almost_equal(ca, np.zeros(3))

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        ca = compact_axis_angle(a)
        assert_array_almost_equal(norm_vector(ca), a[:3])
        assert_almost_equal(np.linalg.norm(ca), a[3])


def test_conversions_compact_axis_angle_quaternion():
    """Test conversions between compact axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = compact_axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, np.array([0, 0, 0]))
    q2 = quaternion_from_compact_axis_angle(a)
    assert_array_almost_equal(q2, q)

    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_compact_axis_angle(random_state)
        q = quaternion_from_compact_axis_angle(a)

        a2 = compact_axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = quaternion_from_compact_axis_angle(a2)
        assert_quaternion_equal(q, q2)


def test_conversions_to_matrix():
    """Test conversions to rotation matrix."""
    R = np.eye(3)
    R2R = matrix_from(R=R)
    assert_array_almost_equal(R2R, R)

    a = np.array([1, 0, 0, 0])
    a2R = matrix_from(a=a)
    assert_array_almost_equal(a2R, R)

    q = np.array([1, 0, 0, 0])
    q2R = matrix_from(q=q)
    assert_array_almost_equal(q2R, R)

    e_xyz = np.array([0, 0, 0])
    e_xyz2R = matrix_from(e_xyz=e_xyz)
    assert_array_almost_equal(e_xyz2R, R)

    e_zyx = np.array([0, 0, 0])
    e_zyx2R = matrix_from(e_zyx=e_zyx)
    assert_array_almost_equal(e_zyx2R, R)

    assert_raises_regexp(ValueError, "no rotation", matrix_from)


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


def test_interpolate_same_axis_angle():
    """Test interpolation between the same axis-angle rotation.

    See issue #45.
    """
    n_steps = 3
    random_state = np.random.RandomState(42)
    a = random_axis_angle(random_state)
    traj = [axis_angle_slerp(a, a, t) for t in np.linspace(0, 1, n_steps)]
    assert_equal(len(traj), n_steps)
    assert_array_almost_equal(traj[0], a)
    assert_array_almost_equal(traj[1], a)
    assert_array_almost_equal(traj[2], a)


def test_interpolate_almost_same_axis_angle():
    """Test interpolation between almost the same axis-angle rotation."""
    n_steps = 3
    random_state = np.random.RandomState(42)
    a1 = random_axis_angle(random_state)
    a2 = np.copy(a1)
    a2[-1] += np.finfo("float").eps
    traj = [axis_angle_slerp(a1, a2, t) for t in np.linspace(0, 1, n_steps)]
    assert_equal(len(traj), n_steps)
    assert_array_almost_equal(traj[0], a1)
    assert_array_almost_equal(traj[1], a1)
    assert_array_almost_equal(traj[2], a2)


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


def test_interpolate_same_quaternion():
    """Test interpolation between the same quaternion rotation.

    See issue #45.
    """
    n_steps = 3
    random_state = np.random.RandomState(42)
    a = random_axis_angle(random_state)
    q = quaternion_from_axis_angle(a)
    traj = [quaternion_slerp(q, q, t) for t in np.linspace(0, 1, n_steps)]
    assert_equal(len(traj), n_steps)
    assert_array_almost_equal(traj[0], q)
    assert_array_almost_equal(traj[1], q)
    assert_array_almost_equal(traj[2], q)


def test_quaternion_conventions():
    """Test conversion of quaternion between wxyz and xyzw."""
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    q_xyzw = quaternion_xyzw_from_wxyz(q_wxyz)
    assert_array_equal(q_xyzw, np.array([0.0, 0.0, 0.0, 1.0]))
    q_wxyz2 = quaternion_wxyz_from_xyzw(q_xyzw)
    assert_array_equal(q_wxyz, q_wxyz2)

    random_state = np.random.RandomState(42)
    q_wxyz_random = random_quaternion(random_state)
    q_xyzw_random = quaternion_xyzw_from_wxyz(q_wxyz_random)
    assert_array_equal(q_xyzw_random[:3], q_wxyz_random[1:])
    assert_equal(q_xyzw_random[3], q_wxyz_random[0])
    q_wxyz_random2 = quaternion_wxyz_from_xyzw(q_xyzw_random)
    assert_array_equal(q_wxyz_random, q_wxyz_random2)


def test_concatenate_quaternions():
    """Test concatenation of two quaternions."""
    # Until ea9adc5, this combination of a list and a numpy array raised
    # a ValueError:
    q1 = [1, 0, 0, 0]
    q2 = np.array([0, 0, 0, 1])
    q12 = concatenate_quaternions(q1, q2)
    assert_array_almost_equal(q12, np.array([0, 0, 0, 1]))

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


def test_quaternion_hamilton():
    """Test if quaternion multiplication follows Hamilton's convention."""
    q_ij = concatenate_quaternions(q_i, q_j)
    assert_array_equal(q_k, q_ij)
    q_ijk = concatenate_quaternions(q_ij, q_k)
    assert_array_equal(-q_id, q_ijk)


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
        q = random_quaternion(random_state)
        v = random_vector(random_state)
        vq = q_prod_vector(q, v)
        vq2 = concatenate_quaternions(concatenate_quaternions(
            q, np.hstack(([0], v))), q_conj(q))[1:]
        assert_array_almost_equal(vq, vq2)


def test_quaternion_invert():
    """Test unit quaternion inversion with conjugate."""
    q = np.array([0.58183503, -0.75119889, -0.24622332, 0.19116072])
    q_inv = q_conj(q)
    q_q_inv = concatenate_quaternions(q, q_inv)
    assert_array_almost_equal(q_id, q_q_inv)


def test_quaternion_gradient_integration():
    """Test integration of quaternion gradients."""
    n_steps = 21
    dt = 0.1
    random_state = np.random.RandomState(3)
    for _ in range(5):
        q1 = random_quaternion(random_state)
        q2 = random_quaternion(random_state)
        Q = np.vstack([quaternion_slerp(q1, q2, t)
                    for t in np.linspace(0, 1, n_steps)])
        angular_velocities = quaternion_gradient(Q, dt)
        Q2 = quaternion_integrate(angular_velocities, q1, dt)
        assert_array_almost_equal(Q, Q2)


def test_quaternion_rotation_consistent_with_multiplication():
    """Test if quaternion rotation and multiplication are Hamiltonian."""
    random_state = np.random.RandomState(1)
    for _ in range(5):
        v = random_vector(random_state)
        q = random_quaternion(random_state)
        v_im = np.hstack(((0.0,), v))
        qv_mult = concatenate_quaternions(
            q, concatenate_quaternions(v_im, q_conj(q)))[1:]
        qv_rot = q_prod_vector(q, v)
        assert_array_almost_equal(qv_mult, qv_rot)


def test_quaternion_dist():
    """Test angular metric of quaternions."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q1 = quaternion_from_axis_angle(random_axis_angle(random_state))
        q2 = quaternion_from_axis_angle(random_axis_angle(random_state))
        q1_to_q1 = quaternion_dist(q1, q1)
        assert_almost_equal(q1_to_q1, 0.0)
        q2_to_q2 = quaternion_dist(q2, q2)
        assert_almost_equal(q2_to_q2, 0.0)
        q1_to_q2 = quaternion_dist(q1, q2)
        q2_to_q1 = quaternion_dist(q2, q1)
        assert_almost_equal(q1_to_q2, q2_to_q1)
        assert_greater(2.0 * np.pi, q1_to_q2)


def test_quaternion_dist_for_identical_rotations():
    """Test angular metric of quaternions q and -q."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q = quaternion_from_axis_angle(random_axis_angle(random_state))
        assert_array_almost_equal(matrix_from_quaternion(q),
                                  matrix_from_quaternion(-q))
        assert_equal(quaternion_dist(q, -q), 0.0)


def test_quaternion_dist_for_almost_identical_rotations():
    """Test angular metric of quaternions q and ca. -q."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        a = random_axis_angle(random_state)
        q1 = quaternion_from_axis_angle(a)
        r = 1e-4 * random_state.randn(4)
        q2 = -quaternion_from_axis_angle(a + r)
        assert_almost_equal(quaternion_dist(q1, q2), 0.0, places=3)


def test_quaternion_diff():
    """Test difference of quaternions."""
    random_state = np.random.RandomState(0)

    for _ in range(5):
        q1 = random_quaternion(random_state)
        q2 = random_quaternion(random_state)
        a_diff = quaternion_diff(q1, q2)          # q1 - q2
        q_diff = quaternion_from_axis_angle(a_diff)
        q3 = concatenate_quaternions(q_diff, q2)  # q1 - q2 + q2
        assert_quaternion_equal(q1, q3)


def test_id_rot():
    """Test equivalence of constants that represent no rotation."""
    assert_array_almost_equal(R_id, matrix_from_axis_angle(a_id))
    assert_array_almost_equal(R_id, matrix_from_quaternion(q_id))
    assert_array_almost_equal(R_id, matrix_from_euler_xyz(e_xyz_id))
    assert_array_almost_equal(R_id, matrix_from_euler_zyx(e_zyx_id))


def test_check_matrix_threshold():
    """Test matrix threshold.

    See issue #54.
    """
    R = np.array([
        [-9.15361835e-01,  4.01808328e-01,  2.57475872e-02],
        [ 5.15480570e-02,  1.80374088e-01, -9.82246499e-01],
        [-3.99318925e-01, -8.97783496e-01, -1.85819250e-01]])
    assert_rotation_matrix(R)
    check_matrix(R)


def test_asssert_rotation_matrix_behaves_like_check_matrix():
    """Test of both checks for rotation matrix validity behave similar."""
    random_state = np.random.RandomState(2345)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)
        original_value = R[2, 2]
        for error in [0, 1e-8, 1e-7, 1e-5, 1e-4, 1]:
            R[2, 2] = original_value + error
            try:
                assert_rotation_matrix(R)
                check_matrix(R)
            except AssertionError:
                assert_raises_regexp(
                    ValueError, "Expected rotation matrix", check_matrix, R)


def test_deactivate_rotation_matrix_precision_error():
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    assert_raises_regexp(
        ValueError, "Expected rotation matrix", check_matrix, R)
    with warnings.catch_warnings(record=True) as w:
        check_matrix(R, strict_check=False)
        assert_equal(len(w), 2)
