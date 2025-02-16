import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import pytransform3d.rotations as pr


def test_check_quaternion():
    """Test input validation for quaternion representation."""
    q_list = [1, 0, 0, 0]
    q = pr.check_quaternion(q_list)
    assert_array_almost_equal(q_list, q)
    assert type(q) is np.ndarray
    assert q.dtype == np.float64

    rng = np.random.default_rng(0)
    q = rng.standard_normal(4)
    q = pr.check_quaternion(q)
    assert pytest.approx(np.linalg.norm(q)) == 1.0

    with pytest.raises(ValueError, match="Expected quaternion with shape"):
        pr.check_quaternion(np.zeros(3))
    with pytest.raises(ValueError, match="Expected quaternion with shape"):
        pr.check_quaternion(np.zeros((3, 3)))

    q = np.array([0.0, 1.2, 0.0, 0.0])
    q2 = pr.check_quaternion(q, unit=False)
    assert_array_almost_equal(q, q2)


def test_check_quaternions():
    """Test input validation for sequence of quaternions."""
    Q_list = [[1, 0, 0, 0]]
    Q = pr.check_quaternions(Q_list)
    assert_array_almost_equal(Q_list, Q)
    assert type(Q) is np.ndarray
    assert Q.dtype == np.float64
    assert Q.ndim == 2
    assert_array_equal(Q.shape, (1, 4))

    Q = np.array([[2, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]])
    Q = pr.check_quaternions(Q)
    for i in range(len(Q)):
        assert pytest.approx(np.linalg.norm(Q[i])) == 1

    with pytest.raises(
        ValueError, match="Expected quaternion array with shape"
    ):
        pr.check_quaternions(np.zeros(4))
    with pytest.raises(
        ValueError, match="Expected quaternion array with shape"
    ):
        pr.check_quaternions(np.zeros((3, 3)))

    Q = np.array([[0.0, 1.2, 0.0, 0.0]])
    Q2 = pr.check_quaternions(Q, unit=False)
    assert_array_almost_equal(Q, Q2)


def test_quaternion_requires_renormalization():
    assert not pr.quaternion_requires_renormalization(pr.q_id)

    q = pr.q_id + np.array([1e-3, 0.0, 0.0, 0.0])
    assert pr.quaternion_requires_renormalization(q)


def test_quaternion_double():
    rng = np.random.default_rng(2235)
    for _ in range(5):
        q1 = pr.random_quaternion(rng)
        q2 = pr.quaternion_double(q1)
        pr.assert_quaternion_equal(q1, q2)


def test_pick_closest_quaternion():
    rng = np.random.default_rng(483)
    for _ in range(10):
        q = pr.random_quaternion(rng)
        assert_array_almost_equal(pr.pick_closest_quaternion(q, q), q)
        assert_array_almost_equal(pr.pick_closest_quaternion(-q, q), q)


def test_quaternion_gradient_integration():
    """Test integration of quaternion gradients."""
    n_steps = 21
    dt = 0.1
    rng = np.random.default_rng(3)
    for _ in range(5):
        q1 = pr.random_quaternion(rng)
        q2 = pr.random_quaternion(rng)
        Q = np.vstack(
            [pr.quaternion_slerp(q1, q2, t) for t in np.linspace(0, 1, n_steps)]
        )
        angular_velocities = pr.quaternion_gradient(Q, dt)
        Q2 = pr.quaternion_integrate(angular_velocities, q1, dt)
        assert_array_almost_equal(Q, Q2)


def test_concatenate_quaternions():
    """Test concatenation of two quaternions."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        q1 = pr.quaternion_from_axis_angle(pr.random_axis_angle(rng))
        q2 = pr.quaternion_from_axis_angle(pr.random_axis_angle(rng))

        R1 = pr.matrix_from_quaternion(q1)
        R2 = pr.matrix_from_quaternion(q2)

        q12 = pr.concatenate_quaternions(q1, q2)
        R12 = np.dot(R1, R2)
        q12R = pr.quaternion_from_matrix(R12)

        pr.assert_quaternion_equal(q12, q12R)


def test_concatenate_quaternions_list_array():
    """Test concatenation of two quaternions given as list and array."""
    # Until ea9adc5, this combination of a list and a numpy array raised
    # a ValueError:
    q1 = [1, 0, 0, 0]
    q2 = np.array([0, 0, 0, 1])
    q12 = pr.concatenate_quaternions(q1, q2)
    assert_array_almost_equal(q12, np.array([0, 0, 0, 1]))


def test_quaternion_hamilton():
    """Test if quaternion multiplication follows Hamilton's convention."""
    q_ij = pr.concatenate_quaternions(pr.q_i, pr.q_j)
    assert_array_equal(pr.q_k, q_ij)
    q_ijk = pr.concatenate_quaternions(q_ij, pr.q_k)
    assert_array_equal(-pr.q_id, q_ijk)


def test_quaternion_rotation():
    """Test quaternion rotation."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = pr.quaternion_from_axis_angle(pr.random_axis_angle(rng))
        R = pr.matrix_from_quaternion(q)
        v = pr.random_vector(rng)
        vR = np.dot(R, v)
        vq = pr.q_prod_vector(q, v)
        assert_array_almost_equal(vR, vq)


def test_quaternion_rotation_consistent_with_multiplication():
    """Test if quaternion rotation and multiplication are Hamiltonian."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        v = pr.random_vector(rng)
        q = pr.random_quaternion(rng)
        v_im = np.hstack(((0.0,), v))
        qv_mult = pr.concatenate_quaternions(
            q, pr.concatenate_quaternions(v_im, pr.q_conj(q))
        )[1:]
        qv_rot = pr.q_prod_vector(q, v)
        assert_array_almost_equal(qv_mult, qv_rot)


def test_quaternion_conjugate():
    """Test quaternion conjugate."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = pr.random_quaternion(rng)
        v = pr.random_vector(rng)
        vq = pr.q_prod_vector(q, v)
        vq2 = pr.concatenate_quaternions(
            pr.concatenate_quaternions(q, np.hstack(([0], v))), pr.q_conj(q)
        )[1:]
        assert_array_almost_equal(vq, vq2)


def test_quaternion_invert():
    """Test unit quaternion inversion with conjugate."""
    q = np.array([0.58183503, -0.75119889, -0.24622332, 0.19116072])
    q_inv = pr.q_conj(q)
    q_q_inv = pr.concatenate_quaternions(q, q_inv)
    assert_array_almost_equal(pr.q_id, q_q_inv)


def test_quaternion_dist():
    """Test angular metric of quaternions."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        q1 = pr.quaternion_from_axis_angle(pr.random_axis_angle(rng))
        q2 = pr.quaternion_from_axis_angle(pr.random_axis_angle(rng))
        q1_to_q1 = pr.quaternion_dist(q1, q1)
        assert pytest.approx(q1_to_q1) == 0.0
        q2_to_q2 = pr.quaternion_dist(q2, q2)
        assert pytest.approx(q2_to_q2) == 0.0
        q1_to_q2 = pr.quaternion_dist(q1, q2)
        q2_to_q1 = pr.quaternion_dist(q2, q1)
        assert pytest.approx(q1_to_q2) == q2_to_q1
        assert 2.0 * np.pi > q1_to_q2


def test_quaternion_dist_for_identical_rotations():
    """Test angular metric of quaternions q and -q."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        q = pr.quaternion_from_axis_angle(pr.random_axis_angle(rng))
        assert_array_almost_equal(
            pr.matrix_from_quaternion(q), pr.matrix_from_quaternion(-q)
        )
        assert pr.quaternion_dist(q, -q) == 0.0


def test_quaternion_dist_for_almost_identical_rotations():
    """Test angular metric of quaternions q and ca. -q."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        a = pr.random_axis_angle(rng)
        q1 = pr.quaternion_from_axis_angle(a)
        r = 1e-4 * rng.standard_normal(4)
        q2 = -pr.quaternion_from_axis_angle(a + r)
        assert pytest.approx(pr.quaternion_dist(q1, q2), abs=1e-3) == 0.0


def test_quaternion_diff():
    """Test difference of quaternions."""
    rng = np.random.default_rng(0)

    for _ in range(5):
        q1 = pr.random_quaternion(rng)
        q2 = pr.random_quaternion(rng)
        a_diff = pr.quaternion_diff(q1, q2)  # q1 - q2
        q_diff = pr.quaternion_from_axis_angle(a_diff)
        q3 = pr.concatenate_quaternions(q_diff, q2)  # q1 - q2 + q2
        pr.assert_quaternion_equal(q1, q3)


def test_quaternion_from_euler():
    """Quaternion from Euler angles."""
    with pytest.raises(
        ValueError, match="Axis index i \\(-1\\) must be in \\[0, 1, 2\\]"
    ):
        pr.quaternion_from_euler(np.zeros(3), -1, 0, 2, True)
    with pytest.raises(
        ValueError, match="Axis index i \\(3\\) must be in \\[0, 1, 2\\]"
    ):
        pr.quaternion_from_euler(np.zeros(3), 3, 0, 2, True)
    with pytest.raises(
        ValueError, match="Axis index j \\(-1\\) must be in \\[0, 1, 2\\]"
    ):
        pr.quaternion_from_euler(np.zeros(3), 2, -1, 2, True)
    with pytest.raises(
        ValueError, match="Axis index j \\(3\\) must be in \\[0, 1, 2\\]"
    ):
        pr.quaternion_from_euler(np.zeros(3), 2, 3, 2, True)
    with pytest.raises(
        ValueError, match="Axis index k \\(-1\\) must be in \\[0, 1, 2\\]"
    ):
        pr.quaternion_from_euler(np.zeros(3), 2, 0, -1, True)
    with pytest.raises(
        ValueError, match="Axis index k \\(3\\) must be in \\[0, 1, 2\\]"
    ):
        pr.quaternion_from_euler(np.zeros(3), 2, 0, 3, True)

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
    rng = np.random.default_rng(83)
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
                e2 = pr.euler_from_quaternion(q, ea[0], ea[1], ea[2], extrinsic)
                q2 = pr.quaternion_from_euler(
                    e2, ea[0], ea[1], ea[2], extrinsic
                )

                pr.assert_quaternion_equal(q, q2)


def test_conversions_matrix_quaternion():
    """Test conversions between rotation matrix and quaternion."""
    R = np.eye(3)
    a = pr.axis_angle_from_matrix(R)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))

    rng = np.random.default_rng(0)
    for _ in range(5):
        q = pr.random_quaternion(rng)
        R = pr.matrix_from_quaternion(q)
        pr.assert_rotation_matrix(R)

        q2 = pr.quaternion_from_matrix(R)
        pr.assert_quaternion_equal(q, q2)

        R2 = pr.matrix_from_quaternion(q2)
        assert_array_almost_equal(R, R2)
        pr.assert_rotation_matrix(R2)


def test_quaternion_conventions():
    """Test conversion of quaternion between wxyz and xyzw."""
    q_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    q_xyzw = pr.quaternion_xyzw_from_wxyz(q_wxyz)
    assert_array_equal(q_xyzw, np.array([0.0, 0.0, 0.0, 1.0]))
    q_wxyz2 = pr.quaternion_wxyz_from_xyzw(q_xyzw)
    assert_array_equal(q_wxyz, q_wxyz2)

    rng = np.random.default_rng(42)
    q_wxyz_random = pr.random_quaternion(rng)
    q_xyzw_random = pr.quaternion_xyzw_from_wxyz(q_wxyz_random)
    assert_array_equal(q_xyzw_random[:3], q_wxyz_random[1:])
    assert q_xyzw_random[3] == q_wxyz_random[0]
    q_wxyz_random2 = pr.quaternion_wxyz_from_xyzw(q_xyzw_random)
    assert_array_equal(q_wxyz_random, q_wxyz_random2)
