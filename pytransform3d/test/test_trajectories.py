import numpy as np
import pytest
import pytransform3d.trajectories as ptr
from pytransform3d.trajectories import (
    invert_transforms, transforms_from_pqs, pqs_from_transforms,
    transforms_from_exponential_coordinates,
    exponential_coordinates_from_transforms,
    pqs_from_dual_quaternions, dual_quaternions_from_pqs,
    batch_concatenate_dual_quaternions, batch_dq_prod_vector,
    transforms_from_dual_quaternions, dual_quaternions_from_transforms,
    concat_one_to_many, concat_many_to_one, mirror_screw_axis_direction,
    screw_parameters_from_dual_quaternions,
    dual_quaternions_from_screw_parameters, dual_quaternions_sclerp,
    concat_dynamic, concat_many_to_many)
from pytransform3d.rotations import (
    quaternion_from_matrix, assert_quaternion_equal, active_matrix_from_angle,
    random_quaternion)
import pytransform3d.transformations as pt
from pytransform3d.transformations import (
    exponential_coordinates_from_transform, translate_transform,
    rotate_transform, random_transform, transform_from_pq,
    concatenate_dual_quaternions, dq_prod_vector,
    assert_unit_dual_quaternion_equal, invert_transform, concat,
    transform_from_exponential_coordinates, dual_quaternion_from_transform,
    screw_parameters_from_dual_quaternion)
from pytransform3d.batch_rotations import norm_vectors
from numpy.testing import assert_array_almost_equal


def test_invert_transforms_0dims():
    rng = np.random.default_rng(0)
    A2B = random_transform(rng)
    B2A = invert_transform(A2B)
    assert_array_almost_equal(B2A, invert_transforms(A2B))


def test_invert_transforms_1dims():
    rng = np.random.default_rng(1)
    A2Bs = np.empty((3, 4, 4))
    B2As = np.empty((3, 4, 4))
    for i in range(len(A2Bs)):
        A2Bs[i] = random_transform(rng)
        B2As[i] = invert_transform(A2Bs[i])
    assert_array_almost_equal(B2As, invert_transforms(A2Bs))


def test_invert_transforms_2dims():
    rng = np.random.default_rng(1)
    A2Bs = np.empty((9, 4, 4))
    B2As = np.empty((9, 4, 4))
    for i in range(len(A2Bs)):
        A2Bs[i] = random_transform(rng)
        B2As[i] = invert_transform(A2Bs[i])
    assert_array_almost_equal(
        B2As.reshape(3, 3, 4, 4), invert_transforms(A2Bs.reshape(3, 3, 4, 4)))


def test_concat_one_to_many():
    rng = np.random.default_rng(482)
    A2B = random_transform(rng)
    B2C = random_transform(rng)
    A2C = concat(A2B, B2C)
    assert_array_almost_equal(A2C, concat_one_to_many(A2B, [B2C])[0])

    B2Cs = [random_transform(rng) for _ in range(5)]
    A2Cs = [concat(A2B, B2C) for B2C in B2Cs]
    assert_array_almost_equal(A2Cs, concat_one_to_many(A2B, B2Cs))


def test_concat_many_to_one():
    rng = np.random.default_rng(482)
    A2B = random_transform(rng)
    B2C = random_transform(rng)
    A2C = concat(A2B, B2C)
    assert_array_almost_equal(A2C, concat_many_to_one([A2B], B2C)[0])

    A2Bs = [random_transform(rng) for _ in range(5)]
    A2Cs = [concat(A2B, B2C) for A2B in A2Bs]
    assert_array_almost_equal(A2Cs, concat_many_to_one(A2Bs, B2C))


def test_concat_dynamic():
    rng = np.random.default_rng(84320)
    n_rotations = 5
    A2Bs = np.stack([random_transform(rng) for _ in range(n_rotations)])
    B2Cs = np.stack([random_transform(rng) for _ in range(n_rotations)])
    A2Cs = concat_dynamic(A2Bs, B2Cs)

    for i in range(len(A2Cs)):
        assert_array_almost_equal(A2Cs[i], concat(A2Bs[i], B2Cs[i]))
        assert_array_almost_equal(A2Cs[i], concat_dynamic(A2Bs[i], B2Cs[i]))
    assert_array_almost_equal(concat_dynamic(A2Bs[0], B2Cs),
                              concat_one_to_many(A2Bs[0], B2Cs))
    assert_array_almost_equal(concat_dynamic(A2Bs, B2Cs[0]),
                              concat_many_to_one(A2Bs, B2Cs[0]))
    with pytest.raises(ValueError, match="Expected ndim 2 or 3"):
        concat_dynamic(A2Bs, B2Cs[np.newaxis])
    with pytest.raises(ValueError, match="Expected ndim 2 or 3"):
        concat_dynamic(A2Bs[np.newaxis], B2Cs)


def test_concat_many_to_many():
    rng = np.random.default_rng(84320)
    n_rotations = 5
    A2Bs = np.stack([random_transform(rng) for _ in range(n_rotations)])
    B2Cs = np.stack([random_transform(rng) for _ in range(n_rotations)])
    A2Cs = ptr.concat_many_to_many(A2Bs, B2Cs)
    for i in range(len(A2Bs)):
        assert_array_almost_equal(A2Cs[i], pt.concat(A2Bs[i], B2Cs[i]))

    with pytest.raises(ValueError):
        ptr.concat_many_to_many(A2Bs, B2Cs[:-1])

    with pytest.raises(ValueError):
        ptr.concat_many_to_many(A2Bs, B2Cs[0])


def test_transforms_from_pqs_0dims():
    rng = np.random.default_rng(0)
    pq = np.empty(7)
    pq[:3] = rng.standard_normal(size=3)
    pq[3:] = random_quaternion(rng)
    A2B = transforms_from_pqs(pq, False)
    assert_array_almost_equal(A2B, transform_from_pq(pq))


def test_transforms_from_pqs_1dim():
    P = np.empty((10, 7))
    rng = np.random.default_rng(0)
    P[:, :3] = rng.standard_normal(size=(len(P), 3))
    P[:, 3:] = norm_vectors(rng.standard_normal(size=(len(P), 4)))

    H = transforms_from_pqs(P)
    P2 = pqs_from_transforms(H)

    assert_array_almost_equal(P[:, :3], H[:, :3, 3])
    assert_array_almost_equal(P[:, :3], P2[:, :3])

    for t in range(len(P)):
        assert_quaternion_equal(P[t, 3:], quaternion_from_matrix(H[t, :3, :3]))
        assert_quaternion_equal(P[t, 3:], P2[t, 3:])


def test_transforms_from_pqs_4dims():
    rng = np.random.default_rng(0)
    P = rng.standard_normal(size=(2, 3, 4, 5, 7))
    P[..., 3:] = norm_vectors(P[..., 3:])

    H = transforms_from_pqs(P)
    P2 = pqs_from_transforms(H)

    assert_array_almost_equal(P[..., :3], H[..., :3, 3])
    assert_array_almost_equal(P[..., :3], P2[..., :3])


def test_transforms_from_exponential_coordinates():
    A2B = np.eye(4)
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    A2B2 = transforms_from_exponential_coordinates(Stheta[np.newaxis])[0]
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates([[Stheta], [Stheta]])[0, 0]
    assert_array_almost_equal(A2B, A2B2)

    A2B = translate_transform(np.eye(4), [1.0, 5.0, 0.0])
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 1.0, 5.0, 0.0])
    A2B2 = transforms_from_exponential_coordinates(Stheta[np.newaxis])[0]
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates([[Stheta], [Stheta]])[0, 0]
    assert_array_almost_equal(A2B, A2B2)

    A2B = rotate_transform(np.eye(4), active_matrix_from_angle(2, 0.5 * np.pi))
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.5 * np.pi, 0.0, 0.0, 0.0])
    A2B2 = transforms_from_exponential_coordinates(Stheta[np.newaxis])[0]
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates([[Stheta], [Stheta]])[0, 0]
    assert_array_almost_equal(A2B, A2B2)

    rng = np.random.default_rng(53)
    for _ in range(5):
        A2B = random_transform(rng)
        Stheta = exponential_coordinates_from_transform(A2B)
        A2B2 = transforms_from_exponential_coordinates(Stheta[np.newaxis])[0]
        assert_array_almost_equal(A2B, A2B2)
        A2B2 = transforms_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(A2B, A2B2)
        A2B2 = transforms_from_exponential_coordinates(
            [[Stheta], [Stheta]])[0, 0]
        assert_array_almost_equal(A2B, A2B2)


def test_exponential_coordinates_from_transforms_0dims():
    rng = np.random.default_rng(842)
    Sthetas = rng.standard_normal(size=6)
    H = transforms_from_exponential_coordinates(Sthetas)
    Sthetas2 = exponential_coordinates_from_transforms(H)
    H2 = transforms_from_exponential_coordinates(Sthetas2)
    assert_array_almost_equal(H, H2)


def test_exponential_coordinates_from_transforms_2dims():
    rng = np.random.default_rng(843)
    Sthetas = rng.standard_normal(size=(4, 4, 6))
    H = transforms_from_exponential_coordinates(Sthetas)
    Sthetas2 = exponential_coordinates_from_transforms(H)
    H2 = transforms_from_exponential_coordinates(Sthetas2)
    assert_array_almost_equal(H, H2)


def test_dual_quaternions_from_pqs_0dims():
    rng = np.random.default_rng(844)
    pq = rng.standard_normal(size=7)
    pq[3:] /= np.linalg.norm(pq[3:], axis=-1)[..., np.newaxis]
    dq = dual_quaternions_from_pqs(pq)
    pq2 = pqs_from_dual_quaternions(dq)
    assert_array_almost_equal(pq[:3], pq2[:3])
    assert_quaternion_equal(pq[3:], pq2[3:])


def test_dual_quaternions_from_pqs_1dim():
    rng = np.random.default_rng(845)
    pqs = rng.standard_normal(size=(20, 7))
    pqs[..., 3:] /= np.linalg.norm(pqs[..., 3:], axis=-1)[..., np.newaxis]
    dqs = dual_quaternions_from_pqs(pqs)
    pqs2 = pqs_from_dual_quaternions(dqs)
    for pq, pq2 in zip(pqs, pqs2):
        assert_array_almost_equal(pq[:3], pq2[:3])
        assert_quaternion_equal(pq[3:], pq2[3:])


def test_dual_quaternions_from_pqs_2dims():
    rng = np.random.default_rng(846)
    pqs = rng.standard_normal(size=(5, 5, 7))
    pqs[..., 3:] /= np.linalg.norm(pqs[..., 3:], axis=-1)[..., np.newaxis]
    dqs = dual_quaternions_from_pqs(pqs)
    pqs2 = pqs_from_dual_quaternions(dqs)
    for pq, pq2 in zip(pqs.reshape(-1, 7), pqs2.reshape(-1, 7)):
        assert_array_almost_equal(pq[:3], pq2[:3])
        assert_quaternion_equal(pq[3:], pq2[3:])


def test_batch_concatenate_dual_quaternions_0dims():
    rng = np.random.default_rng(847)
    pqs = rng.standard_normal(size=(2, 7))
    pqs[..., 3:] /= np.linalg.norm(pqs[..., 3:], axis=-1)[..., np.newaxis]
    dqs = dual_quaternions_from_pqs(pqs)
    dq1 = dqs[0]
    dq2 = dqs[1]

    assert_array_almost_equal(
        batch_concatenate_dual_quaternions(dq1, dq2),
        concatenate_dual_quaternions(dq1, dq2))


def test_batch_concatenate_dual_quaternions_2dims():
    rng = np.random.default_rng(848)
    pqs = rng.standard_normal(size=(2, 2, 2, 7))
    pqs[..., 3:] /= np.linalg.norm(pqs[..., 3:], axis=-1)[..., np.newaxis]
    dqs = dual_quaternions_from_pqs(pqs)
    dqs1 = dqs[0]
    dqs2 = dqs[1]

    dqs_result = batch_concatenate_dual_quaternions(dqs1, dqs2)
    for dq_result, dq1, dq2 in zip(
            dqs_result.reshape(-1, 8), dqs1.reshape(-1, 8),
            dqs2.reshape(-1, 8)):
        assert_array_almost_equal(
            concatenate_dual_quaternions(dq1, dq2),
            dq_result)


def test_batch_dual_quaternion_vector_product_0dims():
    rng = np.random.default_rng(849)
    pq = rng.standard_normal(size=7)
    pq[3:] /= np.linalg.norm(pq[3:], axis=-1)[..., np.newaxis]
    dq = dual_quaternions_from_pqs(pq)
    v = rng.standard_normal(size=3)

    assert_array_almost_equal(
        batch_dq_prod_vector(dq, v), dq_prod_vector(dq, v))


def test_batch_dual_quaternion_vector_product_2dims():
    rng = np.random.default_rng(850)
    pqs = rng.standard_normal(size=(3, 4, 7))
    pqs[..., 3:] /= np.linalg.norm(pqs[..., 3:], axis=-1)[..., np.newaxis]
    dqs = dual_quaternions_from_pqs(pqs)
    V = rng.standard_normal(size=(3, 4, 3))

    V_transformed = batch_dq_prod_vector(dqs, V)
    for v_t, dq, v in zip(V_transformed.reshape(-1, 3), dqs.reshape(-1, 8),
                          V.reshape(-1, 3)):
        assert_array_almost_equal(v_t, dq_prod_vector(dq, v))


def test_batch_conversions_dual_quaternions_transforms_0dims():
    rng = np.random.default_rng(851)
    pq = rng.standard_normal(size=7)
    pq[3:] /= np.linalg.norm(pq[3:], axis=-1)[..., np.newaxis]
    dq = dual_quaternions_from_pqs(pq)

    A2B = transforms_from_dual_quaternions(dq)
    dq2 = dual_quaternions_from_transforms(A2B)
    assert_unit_dual_quaternion_equal(dq, dq2)
    A2B2 = transforms_from_dual_quaternions(dq2)
    assert_array_almost_equal(A2B, A2B2)


def test_batch_conversions_dual_quaternions_transforms_3dims():
    rng = np.random.default_rng(852)
    pqs = rng.standard_normal(size=(3, 4, 5, 7))
    pqs[..., 3:] /= np.linalg.norm(pqs[..., 3:], axis=-1)[..., np.newaxis]
    dqs = dual_quaternions_from_pqs(pqs)

    A2Bs = transforms_from_dual_quaternions(dqs)
    dqs2 = dual_quaternions_from_transforms(A2Bs)
    for dq1, dq2 in zip(dqs.reshape(-1, 8), dqs2.reshape(-1, 8)):
        assert_unit_dual_quaternion_equal(dq1, dq2)
    A2Bs2 = transforms_from_dual_quaternions(dqs2)
    assert_array_almost_equal(A2Bs, A2Bs2)


def test_mirror_screw_axis():
    pose = np.array([[0.10156069, -0.02886784, 0.99441042, 0.6753021],
                     [-0.4892026, -0.87182166, 0.02465395, -0.2085889],
                     [0.86623683, -0.48897203, -0.10266503, 0.30462221],
                     [0.0, 0.0, 0.0, 1.0]])
    exponential_coordinates = exponential_coordinates_from_transform(pose)
    mirror_exponential_coordinates = mirror_screw_axis_direction(
        exponential_coordinates.reshape(1, 6))[0]
    pose2 = transform_from_exponential_coordinates(
        mirror_exponential_coordinates)
    assert_array_almost_equal(pose, pose2)


def test_screw_parameters_from_dual_quaternions():
    case_idx0 = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    case_idx1 = np.array([[1., 0., 0., 0., 0., 0.6, 0.65, 0.7]])

    dqs = np.vstack([case_idx0, case_idx1])
    q, s_axis, h, theta = screw_parameters_from_dual_quaternions(dqs)

    assert_array_almost_equal(q[0], np.zeros(3))
    assert_array_almost_equal(q[0], np.zeros(3))
    assert_array_almost_equal(s_axis[0], np.array([1, 0, 0]))
    assert np.isinf(h[0])
    assert pytest.approx(theta[0]) == 0

    assert_array_almost_equal(q[1], np.zeros(3))
    assert_array_almost_equal(
        s_axis[1], norm_vectors(np.array([1.2, 1.3, 1.4])))
    assert np.isinf(h[1])
    assert pytest.approx(theta[1]) == np.linalg.norm(np.array([1.2, 1.3, 1.4]))

    rng = np.random.default_rng(83343)
    dqs = dual_quaternions_from_transforms(
        np.array([random_transform(rng) for _ in range(10)])).reshape(5, 2, -1)
    qs, s_axes, hs, thetas = screw_parameters_from_dual_quaternions(dqs)
    for q, s_axis, h, theta, dq in zip(
            qs.reshape(10, -1),
            s_axes.reshape(10, -1),
            hs.reshape(10),
            thetas.reshape(10),
            dqs.reshape(10, -1)):
        q2, s_axis2, h2, theta2 = screw_parameters_from_dual_quaternion(dq)
        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(s_axis, s_axis2)
        assert pytest.approx(h) == h2
        assert pytest.approx(theta) == theta2

    q, s_axis, h, theta = screw_parameters_from_dual_quaternions(case_idx0)
    q2, s_axis2, h2, theta2 = screw_parameters_from_dual_quaternion(case_idx0)
    assert_array_almost_equal(q, q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert pytest.approx(h) == h2
    assert pytest.approx(theta) == theta2


def test_dual_quaternions_from_screw_parameters():
    q_0 = np.zeros(3)
    s_axis_0 = np.array([1, 0, 0])
    h_0 = np.inf
    theta_0 = 0.0

    q_1 = np.zeros(3)
    s_axis_1 = np.array([0.55297409, 0.57701644, 0.6010588])
    h_1 = np.inf
    theta_1 = 3.6

    q_2 = np.zeros(3)
    s_axis_2 = np.array([0.55396089, 0.5770426, 0.6001243])
    h_2 = 0.0
    theta_2 = 4.1

    qs = np.vstack([q_0, q_1, q_2])
    s_axis = np.vstack([s_axis_0, s_axis_1, s_axis_2])
    hs = np.array([h_0, h_1, h_2])
    thetas = np.array([theta_0, theta_1, theta_2])

    dqs = dual_quaternions_from_screw_parameters(qs, s_axis, hs, thetas)
    pqs = pqs_from_dual_quaternions(dqs)

    assert_array_almost_equal(dqs[0], np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    assert_array_almost_equal(pqs[1], np.r_[s_axis[1] * thetas[1], 1, 0, 0, 0])
    assert_array_almost_equal(pqs[2, :3], [0, 0, 0])

    rng = np.random.default_rng(83323)
    dqs = np.array([ptr.dual_quaternions_from_transforms(
        pt.random_transform(rng)) for _ in range(18)])

    # 1D
    screw_parameters = ptr.screw_parameters_from_dual_quaternions(dqs[0])
    pt.assert_unit_dual_quaternion_equal(
        dqs[0], ptr.dual_quaternions_from_screw_parameters(*screw_parameters))

    # 2D
    screw_parameters = ptr.screw_parameters_from_dual_quaternions(dqs)
    dqs2 = ptr.dual_quaternions_from_screw_parameters(*screw_parameters)
    for dq1, dq2 in zip(dqs, dqs2):
        pt.assert_unit_dual_quaternion_equal(dq1, dq2)

    # 3D
    dqs = dqs.reshape(6, 3, 8)
    screw_parameters = ptr.screw_parameters_from_dual_quaternions(dqs)
    dqs2 = ptr.dual_quaternions_from_screw_parameters(*screw_parameters)
    for dq1, dq2 in zip(dqs.reshape(-1, 8), dqs2.reshape(-1, 8)):
        pt.assert_unit_dual_quaternion_equal(dq1, dq2)


def test_dual_quaternions_sclerp_same_dual_quaternions():
    rng = np.random.default_rng(19)

    pose0 = random_transform(rng)
    dq0 = dual_quaternion_from_transform(pose0)
    pose1 = random_transform(rng)
    dq1 = dual_quaternion_from_transform(pose1)
    dqs = np.vstack([dq0, dq1])

    ts = np.array([0.5, 0.8])

    dqs_res = dual_quaternions_sclerp(dqs, dqs, ts)

    assert_array_almost_equal(dqs, dqs_res)

    with pytest.raises(ValueError, match="must have the same shape"):
        dual_quaternions_sclerp(dqs, dqs[:-1], ts)

    with pytest.raises(ValueError, match="same number of elements"):
        dual_quaternions_sclerp(dqs, dqs, ts[:-1])

    with pytest.raises(ValueError, match="same number of elements"):
        dual_quaternions_sclerp(dqs, dqs, ts[0])


def test_dual_quaternions_sclerp():
    rng = np.random.default_rng(4832238)
    Ts = np.array([pt.random_transform(rng) for _ in range(20)])

    # 3D
    dqs_start = ptr.dual_quaternions_from_transforms(Ts).reshape(5, 4, -1)
    dqs_end = ptr.dual_quaternions_from_transforms(Ts).reshape(5, 4, -1)
    ts = np.linspace(0, 1, 20).reshape(5, 4)
    dqs_int = ptr.dual_quaternions_sclerp(dqs_start, dqs_end, ts)
    assert dqs_int.shape == (5, 4, 8)
    for dq_start, dq_end, t, dq_int in zip(
            dqs_start.reshape(-1, 8),
            dqs_end.reshape(-1, 8),
            ts.reshape(-1),
            dqs_int.reshape(-1, 8)):
        pt.assert_unit_dual_quaternion_equal(
            dq_int, pt.dual_quaternion_sclerp(dq_start, dq_end, t))

    # 1D
    pt.assert_unit_dual_quaternion_equal(
        pt.dual_quaternion_sclerp(dqs_start[0, 0], dqs_end[0, 0], ts[0, 0]),
        ptr.dual_quaternions_sclerp(dqs_start[0, 0], dqs_end[0, 0], ts[0, 0]))


def test_batch_dq_q_conj():
    rng = np.random.default_rng(48338)
    Ts = np.array([pt.random_transform(rng) for _ in range(20)])
    dqs = ptr.dual_quaternions_from_transforms(Ts).reshape(5, 4, -1)

    # 1D
    assert_array_almost_equal(
        pt.transform_from_dual_quaternion(
            pt.concatenate_dual_quaternions(
                dqs[0, 0], ptr.batch_dq_q_conj(dqs[0, 0]))),
        np.eye(4)
    )

    # 3D
    dqs_inv = ptr.batch_dq_q_conj(dqs)
    dqs_id = ptr.batch_concatenate_dual_quaternions(dqs, dqs_inv)
    Is = ptr.transforms_from_dual_quaternions(dqs_id)
    for I in Is.reshape(-1, 4, 4):
        assert_array_almost_equal(I, np.eye(4))
