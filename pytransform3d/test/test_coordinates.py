import numpy as np
import pytransform3d.coordinates as pc
from numpy.testing import assert_array_almost_equal


def test_cylindrical_from_cartesian_edge_cases():
    q = pc.cylindrical_from_cartesian(np.zeros(3))
    assert_array_almost_equal(q, np.zeros(3))


def test_cartesian_from_cylindrical_edge_cases():
    q = pc.cartesian_from_cylindrical(np.zeros(3))
    assert_array_almost_equal(q, np.zeros(3))


def test_convert_cartesian_cylindrical():
    rng = np.random.default_rng(0)
    for i in range(1000):
        rho = rng.standard_normal()
        phi = rng.random() * 4 * np.pi - 2 * np.pi
        z = rng.standard_normal()
        p = np.array([rho, phi, z])
        q = pc.cartesian_from_cylindrical(p)
        r = pc.cylindrical_from_cartesian(q)
        assert 0 <= r[0]
        assert -np.pi <= r[1] <= np.pi
        s = pc.cartesian_from_cylindrical(r)
        assert_array_almost_equal(q, s)


def test_spherical_from_cartesian_edge_cases():
    q = pc.spherical_from_cartesian(np.zeros(3))
    assert_array_almost_equal(q, np.zeros(3))


def test_cartesian_from_spherical_edge_cases():
    q = pc.cartesian_from_spherical(np.zeros(3))
    assert_array_almost_equal(q, np.zeros(3))


def test_convert_cartesian_spherical():
    rng = np.random.default_rng(1)
    for i in range(1000):
        rho = rng.standard_normal()
        theta = rng.random() * 4 * np.pi - 2 * np.pi
        phi = rng.random() * 4 * np.pi - 2 * np.pi
        p = np.array([rho, theta, phi])
        q = pc.cartesian_from_spherical(p)
        r = pc.spherical_from_cartesian(q)
        assert 0 <= r[0]
        assert 0 <= r[1] <= np.pi
        assert -np.pi <= r[2] <= np.pi
        s = pc.cartesian_from_spherical(r)
        assert_array_almost_equal(q, s)


def test_spherical_from_cylindrical_edge_cases():
    q = pc.spherical_from_cylindrical(np.zeros(3))
    assert_array_almost_equal(q, np.zeros(3))


def test_cylindrical_from_spherical_edge_cases():
    q = pc.cylindrical_from_spherical(np.zeros(3))
    assert_array_almost_equal(q, np.zeros(3))


def test_convert_cylindrical_spherical():
    rng = np.random.default_rng(2)
    for i in range(1000):
        rho = rng.standard_normal()
        theta = rng.random() * 4 * np.pi - 2 * np.pi
        phi = rng.random() * 4 * np.pi - 2 * np.pi
        p = np.array([rho, theta, phi])
        q = pc.cylindrical_from_spherical(p)
        r = pc.spherical_from_cylindrical(q)
        s = pc.cylindrical_from_spherical(r)
        assert_array_almost_equal(q, s)


def test_integer_inputs():
    assert_array_almost_equal(
        pc.spherical_from_cylindrical([1, 0, 0]),
        pc.spherical_from_cylindrical([1.0, 0.0, 0.0]),
    )
    assert_array_almost_equal(
        pc.spherical_from_cartesian([1, 0, 0]),
        pc.spherical_from_cartesian([1.0, 0.0, 0.0]),
    )
    assert_array_almost_equal(
        pc.cylindrical_from_cartesian([0, 1, 0]),
        pc.cylindrical_from_cartesian([0.0, 1.0, 0.0]),
    )
    assert_array_almost_equal(
        pc.cartesian_from_cylindrical([1, 1, 0]),
        pc.cartesian_from_cylindrical([1.0, 1.0, 0.0]),
    )
    assert_array_almost_equal(
        pc.cartesian_from_spherical([1, 1, 0]),
        pc.cartesian_from_spherical([1.0, 1.0, 0.0]),
    )
    assert_array_almost_equal(
        pc.cylindrical_from_spherical([1, 1, 0]),
        pc.cylindrical_from_spherical([1.0, 1.0, 0.0]),
    )
