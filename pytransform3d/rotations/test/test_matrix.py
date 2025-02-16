import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

import pytransform3d.rotations as pr


def test_check_matrix():
    """Test input validation for rotation matrix."""
    R_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    R = pr.check_matrix(R_list)
    assert type(R) is np.ndarray
    assert R.dtype == np.float64

    R_int_array = np.eye(3, dtype=int)
    R = pr.check_matrix(R_int_array)
    assert type(R) is np.ndarray
    assert R.dtype == np.float64

    R_array = np.eye(3)
    R = pr.check_matrix(R_array)
    assert_array_equal(R_array, R)

    R = np.eye(4)
    with pytest.raises(ValueError, match="Expected rotation matrix with shape"):
        pr.check_matrix(R)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0.1, 1]])
    with pytest.raises(ValueError, match="inversion by transposition"):
        pr.check_matrix(R)
    with warnings.catch_warnings(record=True) as w:
        pr.check_matrix(R, strict_check=False)
        assert len(w) == 1

    R = np.array([[1, 0, 1e-16], [0, 1, 0], [0, 0, 1]])
    R2 = pr.check_matrix(R)
    assert_array_equal(R, R2)

    R = -np.eye(3)
    with pytest.raises(ValueError, match="determinant"):
        pr.check_matrix(R)
    with warnings.catch_warnings(record=True) as w:
        pr.check_matrix(R, strict_check=False)
        assert len(w) == 1


def test_check_matrix_threshold():
    """Test matrix threshold.

    See issue #54.
    """
    R = np.array(
        [
            [-9.15361835e-01, 4.01808328e-01, 2.57475872e-02],
            [5.15480570e-02, 1.80374088e-01, -9.82246499e-01],
            [-3.99318925e-01, -8.97783496e-01, -1.85819250e-01],
        ]
    )
    pr.assert_rotation_matrix(R)
    pr.check_matrix(R)


def test_assert_rotation_matrix_behaves_like_check_matrix():
    """Test of both checks for rotation matrix validity behave similar."""
    rng = np.random.default_rng(2345)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        R = pr.matrix_from_axis_angle(a)
        original_value = R[2, 2]
        for error in [0, 1e-8, 1e-7, 1e-5, 1e-4, 1]:
            R[2, 2] = original_value + error
            try:
                pr.assert_rotation_matrix(R)
                pr.check_matrix(R)
            except AssertionError:
                with pytest.raises(
                    ValueError, match="Expected rotation matrix"
                ):
                    pr.check_matrix(R)


def test_deactivate_rotation_matrix_precision_error():
    R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    with pytest.raises(ValueError, match="Expected rotation matrix"):
        pr.check_matrix(R)
    with warnings.catch_warnings(record=True) as w:
        pr.check_matrix(R, strict_check=False)
        assert len(w) == 1


def test_matrix_requires_renormalization():
    R = np.eye(3)
    assert not pr.matrix_requires_renormalization(R)

    R[1, 0] += 1e-3
    assert pr.matrix_requires_renormalization(R)

    rng = np.random.default_rng(39232)
    R_total = np.eye(3)
    for _ in range(10):
        e = pr.random_vector(rng, 3)
        R = pr.active_matrix_from_extrinsic_roll_pitch_yaw(e)
        assert not pr.matrix_requires_renormalization(R, tolerance=1e-16)
        R_total = np.dot(R, R_total)
    assert pr.matrix_requires_renormalization(R_total, tolerance=1e-16)


def test_norm_rotation_matrix():
    R = pr.norm_matrix(np.eye(3))
    assert_array_equal(R, np.eye(3))

    R[1, 0] += np.finfo(float).eps
    R = pr.norm_matrix(R)
    assert_array_equal(R, np.eye(3))
    assert np.linalg.det(R) == 1.0

    R = np.eye(3)
    R[1, 1] += 0.3
    R_norm = pr.norm_matrix(R)
    assert pytest.approx(np.linalg.det(R_norm)) == 1.0
    assert_array_almost_equal(np.eye(3), R_norm)


def test_matrix_from_two_vectors():
    with pytest.raises(ValueError, match="a must not be the zero vector"):
        pr.matrix_from_two_vectors(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="b must not be the zero vector"):
        pr.matrix_from_two_vectors(np.ones(3), np.zeros(3))
    with pytest.raises(ValueError, match="a and b must not be parallel"):
        pr.matrix_from_two_vectors(np.ones(3), np.ones(3))

    R = pr.matrix_from_two_vectors(pr.unitx, pr.unity)
    assert_array_almost_equal(R, np.eye(3))

    rng = np.random.default_rng(28)
    for _ in range(5):
        a = pr.random_vector(rng, 3)
        b = pr.random_vector(rng, 3)
        R = pr.matrix_from_two_vectors(a, b)
        pr.assert_rotation_matrix(R)
        assert_array_almost_equal(pr.norm_vector(a), R[:, 0])
        assert (
            pytest.approx(pr.angle_between_vectors(b, R[:, 2])) == 0.5 * np.pi
        )


def test_conversions_matrix_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    R = np.eye(3)
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, -np.pi, 0.0])
    )
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, 0.0, -np.pi])
    )
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([0, 1, 0, np.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([0.0, -np.pi, -np.pi])
    )
    a = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, np.pi]))

    a = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, np.pi])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a2, a)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = pr.random_axis_angle(rng)
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

    R = pr.passive_matrix_from_angle(2, np.pi)
    assert pytest.approx(np.trace(R)) == -1
    a = pr.axis_angle_from_matrix(R)
    axis = (
        1.0
        / np.sqrt(2.0 * (1 + R[2, 2]))
        * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
    )
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    R = pr.passive_matrix_from_angle(1, np.pi)
    assert pytest.approx(np.trace(R)) == -1
    a = pr.axis_angle_from_matrix(R)
    axis = (
        1.0
        / np.sqrt(2.0 * (1 + R[1, 1]))
        * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
    )
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    R = pr.passive_matrix_from_angle(0, np.pi)
    assert pytest.approx(np.trace(R)) == -1
    a = pr.axis_angle_from_matrix(R)
    axis = (
        1.0
        / np.sqrt(2.0 * (1 + R[0, 0]))
        * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
    )
    pr.assert_axis_angle_equal(a, np.hstack((axis, (np.pi,))))

    # normal case is omitted here


def test_conversions_matrix_compact_axis_angle():
    """Test conversions between rotation matrix and axis-angle."""
    R = np.eye(3)
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.zeros(3))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, -np.pi, 0.0])
    )
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.array([0, 0, np.pi]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([-np.pi, 0.0, -np.pi])
    )
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.array([0, np.pi, 0]))

    R = pr.active_matrix_from_intrinsic_euler_xyz(
        np.array([0.0, -np.pi, -np.pi])
    )
    a = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a, np.array([np.pi, 0, 0]))

    a = np.array([np.sqrt(0.5) * np.pi, np.sqrt(0.5) * np.pi, 0.0])
    R = pr.matrix_from_compact_axis_angle(a)
    a2 = pr.compact_axis_angle_from_matrix(R)
    pr.assert_compact_axis_angle_equal(a2, a)

    rng = np.random.default_rng(0)
    for _ in range(50):
        a = pr.random_compact_axis_angle(rng)
        R = pr.matrix_from_compact_axis_angle(a)
        pr.assert_rotation_matrix(R)

        a2 = pr.compact_axis_angle_from_matrix(R)
        pr.assert_compact_axis_angle_equal(a, a2)

        R2 = pr.matrix_from_compact_axis_angle(a2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_issue43():
    """Test axis_angle_from_matrix() with angles close to 0 and pi."""
    a = np.array([-1.0, 1.0, 1.0, np.pi - 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1.0, 1.0, 1.0, 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1.0, 1.0, 1.0, np.pi + 5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)

    a = np.array([-1.0, 1.0, 1.0, -5e-8])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    pr.assert_axis_angle_equal(a, a2)


def test_issue43_numerical_precision():
    """Test numerical precision of angles close to 0 and pi."""
    a = np.array([1.0, 1.0, 1.0, np.pi - 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert axis_dist < 1e-10
    assert abs(a[3] - a2[3]) < 1e-8

    a = np.array([1.0, 1.0, 1.0, 1e-7])
    a[:3] = a[:3] / np.linalg.norm(a[:3])
    R = pr.matrix_from_axis_angle(a)
    a2 = pr.axis_angle_from_matrix(R)
    axis_dist = np.linalg.norm(a[:3] - a2[:3])
    assert axis_dist < 1e-10
    assert abs(a[3] - a2[3]) < 1e-8


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


def test_matrix_from_quaternion_hamilton():
    """Test if the conversion from quaternion to matrix is Hamiltonian."""
    q = np.sqrt(0.5) * np.array([1, 0, 0, 1])
    R = pr.matrix_from_quaternion(q)
    assert_array_almost_equal(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), R)


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

    R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    with pytest.raises(ValueError, match="Expected rotation matrix"):
        pr.quaternion_from_matrix(R)

    R = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.00000001, 1.0], [0.0, 1.0, -0.00000001]]
    )
    q_from_R = pr.quaternion_from_matrix(R)


def test_quaternion_from_matrix_180_not_axis_aligned():
    """Test for bug in rotation by 180 degrees around arbitrary axes."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        a = pr.random_axis_angle(rng)
        a[3] = np.pi
        q = pr.quaternion_from_axis_angle(a)
        R = pr.matrix_from_axis_angle(a)
        q_from_R = pr.quaternion_from_matrix(R)
        pr.assert_quaternion_equal(q, q_from_R)


def test_axis_angle_from_matrix_cos_angle_greater_1():
    R = np.array(
        [
            [
                1.0000000000000004,
                -1.4402617650886727e-08,
                2.3816502339526408e-08,
            ],
            [
                1.4402617501592725e-08,
                1.0000000000000004,
                1.2457848566326355e-08,
            ],
            [
                -2.3816502529500374e-08,
                -1.2457848247850049e-08,
                0.9999999999999999,
            ],
        ]
    )
    a = pr.axis_angle_from_matrix(R)
    assert not any(np.isnan(a))


def test_axis_angle_from_matrix_without_check():
    R = -np.eye(3)
    with warnings.catch_warnings(record=True) as w:
        a = pr.axis_angle_from_matrix(R, check=False)
    assert len(w) == 1
    assert all(np.isnan(a[:3]))


def test_bug_189():
    """Test bug #189"""
    R = np.array(
        [
            [
                -1.0000000000000004e00,
                2.8285718503485576e-16,
                1.0966597378775709e-16,
            ],
            [
                1.0966597378775709e-16,
                -2.2204460492503131e-16,
                1.0000000000000002e00,
            ],
            [
                2.8285718503485576e-16,
                1.0000000000000002e00,
                -2.2204460492503131e-16,
            ],
        ]
    )
    a1 = pr.compact_axis_angle_from_matrix(R)
    a2 = pr.compact_axis_angle_from_matrix(pr.norm_matrix(R))
    assert_array_almost_equal(a1, a2)


def test_bug_198():
    """Test bug #198"""
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    a = pr.compact_axis_angle_from_matrix(R)
    R2 = pr.matrix_from_compact_axis_angle(a)
    assert_array_almost_equal(R, R2)
