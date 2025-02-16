import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_check_axis_angle():
    """Test input validation for axis-angle representation."""
    a_list = [1, 0, 0, 0]
    a = pr.check_axis_angle(a_list)
    assert_array_almost_equal(a_list, a)
    assert type(a) is np.ndarray
    assert a.dtype == np.float64

    rng = np.random.default_rng(0)
    a = np.empty(4)
    a[:3] = pr.random_vector(rng, 3)
    a[3] = rng.standard_normal() * 4.0 * np.pi
    a2 = pr.check_axis_angle(a)
    pr.assert_axis_angle_equal(a, a2)
    assert pytest.approx(np.linalg.norm(a2[:3])) == 1.0
    assert a2[3] > 0
    assert np.pi > a2[3]

    with pytest.raises(
        ValueError, match="Expected axis and angle in array with shape"
    ):
        pr.check_axis_angle(np.zeros(3))
    with pytest.raises(
        ValueError, match="Expected axis and angle in array with shape"
    ):
        pr.check_axis_angle(np.zeros((3, 3)))


def test_check_compact_axis_angle():
    """Test input validation for compact axis-angle representation."""
    a_list = [0, 0, 0]
    a = pr.check_compact_axis_angle(a_list)
    assert_array_almost_equal(a_list, a)
    assert type(a) is np.ndarray
    assert a.dtype == np.float64

    rng = np.random.default_rng(0)
    a = pr.norm_vector(pr.random_vector(rng, 3))
    a *= np.pi + rng.standard_normal() * 4.0 * np.pi
    a2 = pr.check_compact_axis_angle(a)
    pr.assert_compact_axis_angle_equal(a, a2)
    assert np.pi > np.linalg.norm(a2) > 0

    with pytest.raises(
        ValueError, match="Expected axis and angle in array with shape"
    ):
        pr.check_compact_axis_angle(np.zeros(4))
    with pytest.raises(
        ValueError, match="Expected axis and angle in array with shape"
    ):
        pr.check_compact_axis_angle(np.zeros((3, 3)))


def test_norm_axis_angle():
    """Test normalization of angle-axis representation."""
    a = np.array([1.0, 0.0, 0.0, np.pi])
    n = pr.norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 1.0, 0.0, np.pi])
    n = pr.norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, 1.0, np.pi])
    n = pr.norm_axis_angle(a)
    assert_array_almost_equal(a, n)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        angle = rng.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a[3] = angle
        n = pr.norm_axis_angle(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a[3] = angle + angle_offset
            n2 = pr.norm_axis_angle(a)
            assert_array_almost_equal(n, n2)


def test_compact_axis_angle_near_pi():
    assert pr.compact_axis_angle_near_pi(
        np.pi * pr.norm_vector([0.2, 0.1, -0.3])
    )
    assert pr.compact_axis_angle_near_pi(
        (1e-7 + np.pi) * pr.norm_vector([0.2, 0.1, -0.3])
    )
    assert pr.compact_axis_angle_near_pi(
        (-1e-7 + np.pi) * pr.norm_vector([0.2, 0.1, -0.3])
    )
    assert not pr.compact_axis_angle_near_pi(
        (-1e-5 + np.pi) * pr.norm_vector([0.2, 0.1, -0.3])
    )


def test_norm_compact_axis_angle():
    """Test normalization of compact angle-axis representation."""
    a = np.array([np.pi, 0.0, 0.0])
    n = pr.norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, np.pi, 0.0])
    n = pr.norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    a = np.array([0.0, 0.0, np.pi])
    n = pr.norm_compact_axis_angle(a)
    assert_array_almost_equal(a, n)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_compact_axis_angle(rng)
        axis = a / np.linalg.norm(a)
        angle = rng.uniform(-20.0, -10.0 + 2.0 * np.pi)
        a = axis * angle
        n = pr.norm_compact_axis_angle(a)
        for angle_offset in np.arange(0.0, 10.1 * np.pi, 2.0 * np.pi):
            a = axis * (angle + angle_offset)
            n2 = pr.norm_compact_axis_angle(a)
            assert_array_almost_equal(n, n2)


def test_axis_angle_from_two_direction_vectors():
    """Test calculation of axis-angle from two direction vectors."""
    d1 = np.array([1.0, 0.0, 0.0])
    a = pr.axis_angle_from_two_directions(d1, d1)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    a = pr.axis_angle_from_two_directions(d1, np.zeros(3))
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    a = pr.axis_angle_from_two_directions(np.zeros(3), d1)
    pr.assert_axis_angle_equal(a, np.array([1, 0, 0, 0]))

    d2 = np.array([0.0, 1.0, 0.0])
    a = pr.axis_angle_from_two_directions(d1, d2)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, 0.5 * np.pi]))

    d3 = np.array([-1.0, 0.0, 0.0])
    a = pr.axis_angle_from_two_directions(d1, d3)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    d4 = np.array([0.0, -1.0, 0.0])
    a = pr.axis_angle_from_two_directions(d2, d4)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, np.pi]))

    a = pr.axis_angle_from_two_directions(d3, d4)
    pr.assert_axis_angle_equal(a, np.array([0, 0, 1, 0.5 * np.pi]))

    rng = np.random.default_rng(323)
    for _ in range(5):
        R = pr.matrix_from_axis_angle(pr.random_axis_angle(rng))
        v1 = pr.random_vector(rng, 3)
        v2 = R.dot(v1)
        a = pr.axis_angle_from_two_directions(v1, v2)
        assert pytest.approx(pr.angle_between_vectors(v1, a[:3])) == 0.5 * np.pi
        assert pytest.approx(pr.angle_between_vectors(v2, a[:3])) == 0.5 * np.pi
        assert_array_almost_equal(v2, pr.matrix_from_axis_angle(a).dot(v1))


def test_axis_angle_from_compact_axis_angle():
    """Test conversion from compact axis-angle representation."""
    ca = [0.0, 0.0, 0.0]
    a = pr.axis_angle_from_compact_axis_angle(ca)
    assert_array_almost_equal(a, np.array([1.0, 0.0, 0.0, 0.0]))

    rng = np.random.default_rng(1)
    for _ in range(5):
        ca = pr.random_compact_axis_angle(rng)
        a = pr.axis_angle_from_compact_axis_angle(ca)
        assert pytest.approx(np.linalg.norm(ca)) == a[3]
        assert_array_almost_equal(ca[:3] / np.linalg.norm(ca), a[:3])


def test_compact_axis_angle():
    """Test conversion to compact axis-angle representation."""
    a = np.array([1.0, 0.0, 0.0, 0.0])
    ca = pr.compact_axis_angle(a)
    assert_array_almost_equal(ca, np.zeros(3))

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        ca = pr.compact_axis_angle(a)
        assert_array_almost_equal(pr.norm_vector(ca), a[:3])
        assert pytest.approx(np.linalg.norm(ca)) == a[3]


def test_conversions_axis_angle_quaternion():
    """Test conversions between axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = pr.axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, np.array([1, 0, 0, 0]))
    q2 = pr.quaternion_from_axis_angle(a)
    assert_array_almost_equal(q2, q)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        q = pr.quaternion_from_axis_angle(a)

        a2 = pr.axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = pr.quaternion_from_axis_angle(a2)
        pr.assert_quaternion_equal(q, q2)


def test_conversions_compact_axis_angle_quaternion():
    """Test conversions between compact axis-angle and quaternion."""
    q = np.array([1, 0, 0, 0])
    a = pr.compact_axis_angle_from_quaternion(q)
    assert_array_almost_equal(a, np.array([0, 0, 0]))
    q2 = pr.quaternion_from_compact_axis_angle(a)
    assert_array_almost_equal(q2, q)

    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.random_compact_axis_angle(rng)
        q = pr.quaternion_from_compact_axis_angle(a)

        a2 = pr.compact_axis_angle_from_quaternion(q)
        assert_array_almost_equal(a, a2)

        q2 = pr.quaternion_from_compact_axis_angle(a2)
        pr.assert_quaternion_equal(q, q2)


def test_mrp_from_axis_angle():
    rng = np.random.default_rng(98343)
    for _ in range(5):
        a = pr.random_axis_angle(rng)
        mrp = pr.mrp_from_axis_angle(a)
        q = pr.quaternion_from_axis_angle(a)
        assert_array_almost_equal(mrp, pr.mrp_from_quaternion(q))

    assert_array_almost_equal(
        [0.0, 0.0, 0.0], pr.mrp_from_axis_angle([1.0, 0.0, 0.0, 0.0])
    )
    assert_array_almost_equal(
        [0.0, 0.0, 0.0], pr.mrp_from_axis_angle([1.0, 0.0, 0.0, 2.0 * np.pi])
    )
    assert_array_almost_equal(
        [1.0, 0.0, 0.0], pr.mrp_from_axis_angle([1.0, 0.0, 0.0, np.pi])
    )
