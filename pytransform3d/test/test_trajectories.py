import numpy as np
from pytransform3d.trajectories import (
    transforms_from_pqs, pqs_from_transforms,
    transforms_from_exponential_coordinates,
    exponential_coordinates_from_transforms)
from pytransform3d.rotations import (
    quaternion_from_matrix, assert_quaternion_equal, active_matrix_from_angle,
    random_quaternion)
from pytransform3d.transformations import (
    exponential_coordinates_from_transform, translate_transform,
    rotate_transform, random_transform, transform_from_pq)
from pytransform3d.batch_rotations import norm_vectors
from numpy.testing import assert_array_almost_equal


def test_transforms_from_pqs_0dims():
    random_state = np.random.RandomState(0)
    pq = np.empty(7)
    pq[:3] = random_state.randn(3)
    pq[3:] = random_quaternion(random_state)
    A2B = transforms_from_pqs(pq, False)
    assert_array_almost_equal(A2B, transform_from_pq(pq))


def test_transforms_from_pqs_1dim():
    P = np.empty((10, 7))
    random_state = np.random.RandomState(0)
    P[:, :3] = random_state.randn(len(P), 3)
    P[:, 3:] = norm_vectors(random_state.randn(len(P), 4))

    H = transforms_from_pqs(P)
    P2 = pqs_from_transforms(H)

    assert_array_almost_equal(P[:, :3], H[:, :3, 3])
    assert_array_almost_equal(P[:, :3], P2[:, :3])

    for t in range(len(P)):
        assert_quaternion_equal(P[t, 3:], quaternion_from_matrix(H[t, :3, :3]))
        assert_quaternion_equal(P[t, 3:], P2[t, 3:])


def test_transforms_from_pqs_4dims():
    random_state = np.random.RandomState(0)
    P = random_state.randn(2, 3, 4, 5, 7)
    P[..., 3:] = norm_vectors(P[..., 3:])

    H = transforms_from_pqs(P)
    P2 = pqs_from_transforms(H)

    assert_array_almost_equal(P[..., :3], H[..., :3, 3])
    assert_array_almost_equal(P[..., :3], P2[..., :3])


def test_transforms_from_exponential_coordinates():
    A2B = np.eye(4)
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    A2B2 = transforms_from_exponential_coordinates([Stheta])[0]
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates([[Stheta], [Stheta]])[0, 0]
    assert_array_almost_equal(A2B, A2B2)

    A2B = translate_transform(np.eye(4), [1.0, 5.0, 0.0])
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 1.0, 5.0, 0.0])
    A2B2 = transforms_from_exponential_coordinates([Stheta])[0]
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates([[Stheta], [Stheta]])[0, 0]
    assert_array_almost_equal(A2B, A2B2)

    A2B = rotate_transform(np.eye(4), active_matrix_from_angle(2, 0.5 * np.pi))
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.5 * np.pi, 0.0, 0.0, 0.0])
    A2B2 = transforms_from_exponential_coordinates([Stheta])[0]
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)
    A2B2 = transforms_from_exponential_coordinates([[Stheta], [Stheta]])[0, 0]
    assert_array_almost_equal(A2B, A2B2)

    random_state = np.random.RandomState(53)
    for _ in range(5):
        A2B = random_transform(random_state)
        Stheta = exponential_coordinates_from_transform(A2B)
        A2B2 = transforms_from_exponential_coordinates([Stheta])[0]
        assert_array_almost_equal(A2B, A2B2)
        A2B2 = transforms_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(A2B, A2B2)
        A2B2 = transforms_from_exponential_coordinates(
            [[Stheta], [Stheta]])[0, 0]
        assert_array_almost_equal(A2B, A2B2)


def test_exponential_coordinates_from_transforms_0dims():
    random_state = np.random.RandomState(842)
    Sthetas = random_state.randn(6)
    H = transforms_from_exponential_coordinates(Sthetas)
    Sthetas2 = exponential_coordinates_from_transforms(H)
    H2 = transforms_from_exponential_coordinates(Sthetas2)
    assert_array_almost_equal(H, H2)


def test_exponential_coordinates_from_transforms_2dims():
    random_state = np.random.RandomState(843)
    Sthetas = random_state.randn(4, 4, 6)
    H = transforms_from_exponential_coordinates(Sthetas)
    Sthetas2 = exponential_coordinates_from_transforms(H)
    H2 = transforms_from_exponential_coordinates(Sthetas2)
    assert_array_almost_equal(H, H2)
