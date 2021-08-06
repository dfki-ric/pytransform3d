import numpy as np
import pytransform3d.coordinates as pc
from nose.tools import assert_less_equal
from numpy.testing import assert_array_almost_equal


def test_convert_cartesian_cylindrical():
    random_state = np.random.RandomState(0)
    for i in range(1000):
        rho = random_state.randn()
        phi = random_state.rand() * 4 * np.pi - 2 * np.pi
        z = random_state.randn()
        p = np.array([rho, phi, z])
        q = pc.cartesian_from_cylindrical(p)
        r = pc.cylindrical_from_cartesian(q)
        assert_less_equal(0, r[0])
        assert_less_equal(-np.pi, r[1])
        assert_less_equal(r[1], np.pi)
        s = pc.cartesian_from_cylindrical(r)
        assert_array_almost_equal(q, s)


def test_convert_cartesian_spherical():
    random_state = np.random.RandomState(1)
    for i in range(1000):
        rho = random_state.randn()
        theta = random_state.rand() * 4 * np.pi - 2 * np.pi
        phi = random_state.rand() * 4 * np.pi - 2 * np.pi
        p = np.array([rho, theta, phi])
        q = pc.cartesian_from_spherical(p)
        r = pc.spherical_from_cartesian(q)
        assert_less_equal(0, r[0])
        assert_less_equal(0, r[1])
        assert_less_equal(r[1], np.pi)
        assert_less_equal(-np.pi, r[2])
        assert_less_equal(r[2], np.pi)
        s = pc.cartesian_from_spherical(r)
        assert_array_almost_equal(q, s)


def test_convert_cylindrical_spherical():
    random_state = np.random.RandomState(2)
    for i in range(1000):
        rho = random_state.randn()
        theta = random_state.rand() * 4 * np.pi - 2 * np.pi
        phi = random_state.rand() * 4 * np.pi - 2 * np.pi
        p = np.array([rho, theta, phi])
        q = pc.cylindrical_from_spherical(p)
        r = pc.spherical_from_cylindrical(q)
        s = pc.cylindrical_from_spherical(r)
        assert_array_almost_equal(q, s)
