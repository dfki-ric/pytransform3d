import numpy as np
from pytransform3d.trajectories import (
    invert_transforms, transforms_from_pqs, pqs_from_transforms,
    transforms_from_exponential_coordinates,
    exponential_coordinates_from_transforms,
    pqs_from_dual_quaternions, dual_quaternions_from_pqs,
    batch_concatenate_dual_quaternions, batch_dq_prod_vector,
    transforms_from_dual_quaternions, dual_quaternions_from_transforms,
    concat_one_to_many, concat_many_to_one, mirror_screw_axis_direction)
from pytransform3d.rotations import (
    quaternion_from_matrix, assert_quaternion_equal, active_matrix_from_angle,
    random_quaternion)
from pytransform3d.transformations import (
    exponential_coordinates_from_transform, translate_transform,
    rotate_transform, random_transform, transform_from_pq,
    concatenate_dual_quaternions, dq_prod_vector,
    assert_unit_dual_quaternion_equal, invert_transform, concat,
    transform_from_exponential_coordinates)
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
