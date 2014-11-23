import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal
from pytransform.rotations import *


def random_vector(random_state=np.random.RandomState(0)):
    return random_state.rand(3)


def test_perpendicular_to_vectors():
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


def test_conversions_matrix_axis_angle():
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        R = matrix_from_axis_angle(a)

        a2 = axis_angle_from_matrix(R)
        assert_array_almost_equal(a, a2)

        R2 = matrix_from_axis_angle(a2)
        assert_array_almost_equal(R, R2)


def test_conversions_matrix_quaternion():
    random_state = np.random.RandomState(0)
    for _ in range(5):
        q = random_quaternion(random_state)
        R = matrix_from_quaternion(q)

        q2 = quaternion_from_matrix(R)
        assert_array_almost_equal(q, q2)

        R2 = matrix_from_quaternion(q2)
        assert_array_almost_equal(R, R2)


def test_conversions_axis_angle_quaternion():
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_axis_angle(random_state)
        q = quaternion_from_axis_angle(a)

        a2 = axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = quaternion_from_axis_angle(a2)
        assert_array_almost_equal(q, q2)


def test_interpolate_axis_angle():
    n_steps = 10
    random_state = np.random.RandomState(1)
    a1 = random_axis_angle(random_state)
    a2 = random_axis_angle(random_state)

    omega = angle_between_vectors(a1[:3], a2[:3])
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
