import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_almost_equal
from pytransform.rotations import *


def random_vector(random_state=np.random.RandomState(0)):
    return random_state.rand(3)


def random_angle_axis(random_state=np.random.RandomState(0)):
    angle = 2 * np.pi * random_state.rand()
    a = np.array([0, 0, 0, angle])
    a[:3] = random_state.rand(3)
    a[:3] /= np.linalg.norm(a[:3])
    return a


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


def test_conversions_axis_angle_quaternion():
    random_state = np.random.RandomState(0)
    for _ in range(5):
        a = random_angle_axis(random_state)
        q = quaternion_from_axis_angle(a)

        a2 = axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = quaternion_from_axis_angle(a2)
        assert_array_almost_equal(q, q2)


def test_interpolate_axis_angle():
    n_steps = 10
    random_state = np.random.RandomState(1)
    a1 = random_angle_axis(random_state)
    a2 = random_angle_axis(random_state)

    traj = [(1 - t) * a1 + t * a2 for t in np.linspace(0, 1, n_steps)]
    for i in range(n_steps):
        traj[i][:3] /= np.linalg.norm(traj[i][:3])

    axis = norm_vector(perpendicular_to_vectors(a1[:3], a2[:3]))
    angle = angle_between_vectors(a1[:3], a2[:3])
    traj2 = []
    for t in np.linspace(0, 1, n_steps):
        inta = np.hstack((axis, (t * angle,)))
        intaxis = matrix_from_axis_angle(inta).dot(a1[:3])
        intangle = (1 - t) * a1[3] + t * a2[3]
        traj2.append(np.hstack((intaxis, (intangle,))))
    print inta
    print a1, "--->", a2
    print "==="
    print a1
    for t in range(n_steps):
        print "===", t
        print traj[t]
        print traj2[t]
        print "==="
    print a2
