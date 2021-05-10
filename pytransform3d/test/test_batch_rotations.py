import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import batch_rotations as pbr
from nose.tools import assert_almost_equal, assert_raises_regexp
from numpy.testing import assert_array_almost_equal


def test_norm_vectors_0dims():
    random_state = np.random.RandomState(8380)
    V = random_state.randn(3)
    V_unit = pbr.norm_vectors(V)
    assert_almost_equal(np.linalg.norm(V_unit), 1.0)


def test_norm_vectors_1dim():
    random_state = np.random.RandomState(8381)
    V = random_state.randn(100, 3)
    V_unit = pbr.norm_vectors(V)
    assert_array_almost_equal(
        np.linalg.norm(V_unit, axis=1), np.ones(len(V)))


def test_norm_vectors_1dim_output_variable():
    random_state = np.random.RandomState(8381)
    V = random_state.randn(100, 3)
    pbr.norm_vectors(V, out=V)
    assert_array_almost_equal(
        np.linalg.norm(V, axis=1), np.ones(len(V)))


def test_norm_vectors_3dims():
    random_state = np.random.RandomState(8382)
    V = random_state.randn(8, 2, 8, 3)
    V_unit = pbr.norm_vectors(V)
    assert_array_almost_equal(
        np.linalg.norm(V_unit, axis=-1), np.ones(V_unit.shape[:-1]))


def test_norm_vectors_zero():
    V = np.zeros((3, 8, 1, 2))
    V_unit = pbr.norm_vectors(V)
    assert_array_almost_equal(V_unit, V)


def test_angles_between_vectors_0dims():
    random_state = np.random.RandomState(228)
    A = random_state.randn(3)
    B = random_state.randn(3)
    angles = pbr.angles_between_vectors(A, B)
    angles2 = pr.angle_between_vectors(A, B)
    assert_array_almost_equal(angles, angles2)


def test_angles_between_vectors_1dim():
    random_state = np.random.RandomState(229)
    A = random_state.randn(100, 3)
    B = random_state.randn(100, 3)
    angles = pbr.angles_between_vectors(A, B)
    angles2 = [pr.angle_between_vectors(a, b) for a, b in zip(A, B)]
    assert_array_almost_equal(angles, angles2)


def test_angles_between_vectors_3dims():
    random_state = np.random.RandomState(230)
    A = random_state.randn(2, 4, 3, 4)
    B = random_state.randn(2, 4, 3, 4)
    angles = pbr.angles_between_vectors(A, B).ravel()
    angles2 = [pr.angle_between_vectors(a, b)
               for a, b in zip(A.reshape(-1, 4), B.reshape(-1, 4))]
    assert_array_almost_equal(angles, angles2)


def test_active_matrices_from_angles_0dims():
    R = pbr.active_matrices_from_angles(0, 0.4)
    assert_array_almost_equal(R, pr.active_matrix_from_angle(0, 0.4))


def test_active_matrices_from_angles_1dim():
    Rs = pbr.active_matrices_from_angles(1, [0.4, 0.5, 0.6])
    assert_array_almost_equal(Rs[0], pr.active_matrix_from_angle(1, 0.4))
    assert_array_almost_equal(Rs[1], pr.active_matrix_from_angle(1, 0.5))
    assert_array_almost_equal(Rs[2], pr.active_matrix_from_angle(1, 0.6))


def test_active_matrices_from_angles_3dims():
    random_state = np.random.RandomState(8383)
    angles = random_state.randn(2, 3, 4)
    Rs = pbr.active_matrices_from_angles(2, angles)
    Rs = Rs.reshape(-1, 3, 3)
    Rs2 = [pr.active_matrix_from_angle(2, angle)
           for angle in angles.reshape(-1)]
    assert_array_almost_equal(Rs, Rs2)


def test_active_matrices_from_angles_3dims_output_variable():
    random_state = np.random.RandomState(8384)
    angles = random_state.randn(2, 3, 4)
    Rs = np.empty((2, 3, 4, 3, 3))
    pbr.active_matrices_from_angles(2, angles, out=Rs)
    Rs = Rs.reshape(-1, 3, 3)
    Rs2 = [pr.active_matrix_from_angle(2, angle)
           for angle in angles.reshape(-1)]
    assert_array_almost_equal(Rs, Rs2)


def test_active_matrices_from_intrinsic_euler_angles_0dims():
    random_state = np.random.RandomState(8383)
    e = random_state.randn(3)
    R = pbr.active_matrices_from_intrinsic_euler_angles(2, 1, 0, e)
    R2 = pr.active_matrix_from_intrinsic_euler_zyx(e)
    assert_array_almost_equal(R, R2)


def test_active_matrices_from_intrinsic_euler_angles_1dim():
    random_state = np.random.RandomState(8384)
    e = random_state.randn(10, 3)
    Rs = pbr.active_matrices_from_intrinsic_euler_angles(2, 1, 0, e)
    for i in range(len(e)):
        Ri = pr.active_matrix_from_intrinsic_euler_zyx(e[i])
        assert_array_almost_equal(Rs[i], Ri)


def test_active_matrices_from_intrinsic_euler_angles_1dim_output_variables():
    random_state = np.random.RandomState(8384)
    e = random_state.randn(10, 3)
    Rs = np.empty((10, 3, 3))
    pbr.active_matrices_from_intrinsic_euler_angles(2, 1, 0, e, out=Rs)
    for i in range(len(e)):
        Ri = pr.active_matrix_from_intrinsic_euler_zyx(e[i])
        assert_array_almost_equal(Rs[i], Ri)


def test_active_matrices_from_intrinsic_euler_angles_3dims():
    random_state = np.random.RandomState(8385)
    e = random_state.randn(2, 3, 4, 3)
    Rs = pbr.active_matrices_from_intrinsic_euler_angles(
        2, 1, 0, e).reshape(-1, 3, 3)
    e = e.reshape(-1, 3)
    for i in range(len(e)):
        Ri = pr.active_matrix_from_intrinsic_euler_zyx(e[i])
        assert_array_almost_equal(Rs[i], Ri)


def test_active_matrices_from_extrinsic_euler_angles_0dims():
    random_state = np.random.RandomState(8383)
    e = random_state.randn(3)
    R = pbr.active_matrices_from_extrinsic_euler_angles(2, 1, 0, e)
    R2 = pr.active_matrix_from_extrinsic_euler_zyx(e)
    assert_array_almost_equal(R, R2)


def test_active_matrices_from_extrinsic_euler_angles_1dim():
    random_state = np.random.RandomState(8384)
    e = random_state.randn(10, 3)
    Rs = pbr.active_matrices_from_extrinsic_euler_angles(2, 1, 0, e)
    for i in range(len(e)):
        Ri = pr.active_matrix_from_extrinsic_euler_zyx(e[i])
        assert_array_almost_equal(Rs[i], Ri)


def test_active_matrices_from_extrinsic_euler_angles_3dim():
    random_state = np.random.RandomState(8385)
    e = random_state.randn(2, 3, 4, 3)
    Rs = pbr.active_matrices_from_extrinsic_euler_angles(
        2, 1, 0, e).reshape(-1, 3, 3)
    e = e.reshape(-1, 3)
    for i in range(len(e)):
        Ri = pr.active_matrix_from_extrinsic_euler_zyx(e[i])
        assert_array_almost_equal(Rs[i], Ri)


def test_active_matrices_from_extrinsic_euler_angles_1dim_output_variable():
    random_state = np.random.RandomState(8385)
    e = random_state.randn(10, 3)
    Rs = np.empty((10, 3, 3))
    pbr.active_matrices_from_extrinsic_euler_angles(2, 1, 0, e, out=Rs)
    for i in range(len(e)):
        Ri = pr.active_matrix_from_extrinsic_euler_zyx(e[i])
        assert_array_almost_equal(Rs[i], Ri)


def test_cross_product_matrix():
    random_state = np.random.RandomState(3820)
    v = random_state.randn(3)
    assert_array_almost_equal(
        pbr.cross_product_matrices(v), pr.cross_product_matrix(v))


def test_cross_product_matrices():
    random_state = np.random.RandomState(3820)
    V = random_state.randn(2, 2, 3, 3)
    V_cpm = pbr.cross_product_matrices(V)
    V_cpm = V_cpm.reshape(-1, 3, 3)
    V_cpm2 = [pr.cross_product_matrix(v) for v in V.reshape(-1, 3)]
    assert_array_almost_equal(V_cpm, V_cpm2)


def test_matrices_from_quaternions():
    random_state = np.random.RandomState(83)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        R = pbr.matrices_from_quaternions(
            q[np.newaxis], normalize_quaternions=False)[0]
        q2 = pr.quaternion_from_matrix(R)
        pr.assert_quaternion_equal(q, q2)

    for _ in range(5):
        q = random_state.randn(4)
        R = pbr.matrices_from_quaternions(
            q[np.newaxis], normalize_quaternions=True)[0]
        q2 = pr.quaternion_from_matrix(R)
        pr.assert_quaternion_equal(q / np.linalg.norm(q), q2)


def test_quaternions_from_matrices():
    random_state = np.random.RandomState(84)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        R = pr.matrix_from_quaternion(q)
        q2 = pbr.quaternions_from_matrices(R[np.newaxis])[0]
        pr.assert_quaternion_equal(q, q2)

    a = np.array([1.0, 0.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices(R[np.newaxis])[0]
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices(R[np.newaxis])[0]
    assert_array_almost_equal(q, q_from_R)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    q = pr.quaternion_from_axis_angle(a)
    R = pr.matrix_from_axis_angle(a)
    q_from_R = pbr.quaternions_from_matrices(R[np.newaxis])[0]
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


def test_axis_angles_from_matrices_0dims():
    random_state = np.random.RandomState(84)
    A = random_state.randn(3)
    A /= np.linalg.norm(A, axis=-1)[..., np.newaxis]
    A *= random_state.rand() * np.pi

    Rs = pbr.matrices_from_compact_axis_angles(A)
    A2 = pbr.axis_angles_from_matrices(Rs)
    A2_compact = A2[:3] * A2[3]
    assert_array_almost_equal(A, A2_compact)


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


def test_axis_angles_from_matrices_output_variable():
    random_state = np.random.RandomState(84)
    A = random_state.randn(2, 3, 3)
    A /= np.linalg.norm(A, axis=-1)[..., np.newaxis]
    A *= random_state.rand(2, 3, 1) * np.pi
    A[0, 0, :] = 0.0

    Rs = np.empty((2, 3, 3, 3))
    pbr.matrices_from_compact_axis_angles(A, out=Rs)
    A2 = np.empty((2, 3, 4))
    pbr.axis_angles_from_matrices(Rs, out=A2)
    A2_compact = A2[..., :3] * A2[..., 3, np.newaxis]
    assert_array_almost_equal(A, A2_compact)


def test_quaternion_slerp_batch_zero_angle():
    random_state = np.random.RandomState(228)
    q = pr.random_quaternion(random_state)
    Q = pbr.quaternion_slerp_batch(q, q, [0.5])
    pr.assert_quaternion_equal(q, Q[0])


def test_quaternion_slerp_batch():
    random_state = np.random.RandomState(229)
    q_start = pr.random_quaternion(random_state)
    q_end = pr.random_quaternion(random_state)
    t = np.linspace(0, 1, 101)
    Q = pbr.quaternion_slerp_batch(q_start, q_end, t)
    for i in range(len(t)):
        qi = pr.quaternion_slerp(q_start, q_end, t[i])
        pr.assert_quaternion_equal(Q[i], qi)


def test_batch_concatenate_quaternions_mismatch():
    Q1 = np.zeros((1, 2, 4))
    Q2 = np.zeros((1, 2, 3, 4))
    assert_raises_regexp(
        ValueError, "Number of dimensions must be the same.",
        pbr.batch_concatenate_quaternions, Q1, Q2)

    Q1 = np.zeros((1, 2, 4, 4))
    Q2 = np.zeros((1, 2, 3, 4))
    assert_raises_regexp(
        ValueError, "Size of dimension 3 does not match",
        pbr.batch_concatenate_quaternions, Q1, Q2)

    Q1 = np.zeros((1, 2, 3, 3))
    Q2 = np.zeros((1, 2, 3, 4))
    assert_raises_regexp(
        ValueError, "Last dimension of first argument does not match.",
        pbr.batch_concatenate_quaternions, Q1, Q2)

    Q1 = np.zeros((1, 2, 3, 4))
    Q2 = np.zeros((1, 2, 3, 3))
    assert_raises_regexp(
        ValueError, "Last dimension of second argument does not match.",
        pbr.batch_concatenate_quaternions, Q1, Q2)


def test_batch_concatenate_quaternions_1d():
    random_state = np.random.RandomState(230)
    q1 = pr.random_quaternion(random_state)
    q2 = pr.random_quaternion(random_state)
    q12 = np.empty(4)
    pbr.batch_concatenate_quaternions(q1, q2, out=q12)
    assert_array_almost_equal(
        q12, pr.concatenate_quaternions(q1, q2))


def test_batch_q_conj_1d():
    random_state = np.random.RandomState(230)
    q = pr.random_quaternion(random_state)
    assert_array_almost_equal(pr.q_conj(q), pbr.batch_q_conj(q))


def test_batch_concatenate_q_conj():
    random_state = np.random.RandomState(231)
    Q = np.array([pr.random_quaternion(random_state)
                  for _ in range(10)])
    Q = Q.reshape(2, 5, 4)

    Q_conj = pbr.batch_q_conj(Q)
    Q_Q_conj = pbr.batch_concatenate_quaternions(Q, Q_conj)

    assert_array_almost_equal(
        Q_Q_conj.reshape(-1, 4),
        np.array([[1, 0, 0, 0]] * 10))
