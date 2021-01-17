import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import batch_rotations as pbr
from numpy.testing import assert_array_almost_equal


def test_quaternions_from_matrices():
    random_state = np.random.RandomState(84)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        R = pr.matrix_from_quaternion(q)
        q2 = pbr.quaternions_from_matrices([R])[0]
        pr.assert_quaternion_equal(q, q2)

    a = np.array([1.0, 0.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices([R])[0]
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices([R])[0]
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices([R])[0]
    assert_array_almost_equal(q, q_from_R)


def test_quaternions_from_matrices_no_batch():
    random_state = np.random.RandomState(85)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        R = pr.matrix_from_quaternion(q)
        q2 = pbr.quaternions_from_matrices(R)
        pr.assert_quaternion_equal(q, q2)

    a = np.array([1.0, 0.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices(R)
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices(R)
    assert_array_almost_equal(q, q_from_R)


def test_quaternions_from_matrices_4d():
    random_state = np.random.RandomState(84)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        R = pr.matrix_from_quaternion(q)
        q2 = pbr.quaternions_from_matrices([[R, R], [R, R]])
        pr.assert_quaternion_equal(q, q2[0, 0])
        pr.assert_quaternion_equal(q, q2[0, 1])
        pr.assert_quaternion_equal(q, q2[1, 0])
        pr.assert_quaternion_equal(q, q2[1, 1])


def test_axis_angles_from_matrices():
    random_state = np.random.RandomState(84)
    A = random_state.randn(2, 3, 3)
    A /= np.linalg.norm(A, axis=-1)[..., np.newaxis]
    A *= random_state.rand(2, 3, 1) * np.pi
    A[0, 0, :] = 0.0

    Rs = pbr.matrices_from_compact_axis_angles(A)
    A2 = pbr.axis_angles_from_matrices(Rs)
    A2_compact = A2[..., :3] * A2[..., 3, np.newaxis]
    assert_array_almost_equal(A, A2_compact)
