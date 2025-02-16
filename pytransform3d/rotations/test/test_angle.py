import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

import pytransform3d.rotations as pr


def test_norm_angle():
    """Test normalization of angle."""
    rng = np.random.default_rng(0)
    a_norm = rng.uniform(-np.pi, np.pi, size=(100,))
    for b in np.linspace(-10.0 * np.pi, 10.0 * np.pi, 11):
        a = a_norm + b
        assert_array_almost_equal(pr.norm_angle(a), a_norm)

    assert pytest.approx(pr.norm_angle(-np.pi)) == np.pi
    assert pytest.approx(pr.norm_angle(np.pi)) == np.pi


def test_norm_angle_precision():
    # NOTE: it would be better if angles are divided into 1e16 numbers
    #       to test precision of float64 but it is limited by memory
    a_norm = np.linspace(
        np.pi, -np.pi, num=1000000, endpoint=False, dtype=np.float64
    )[::-1]
    for b in np.linspace(-10.0 * np.pi, 10.0 * np.pi, 11):
        a = a_norm + b
        assert_array_almost_equal(pr.norm_angle(a), a_norm)

    # eps and epsneg around zero
    a_eps = np.array([np.finfo(np.float64).eps, -np.finfo(np.float64).eps])
    a_epsneg = np.array(
        [np.finfo(np.float64).epsneg, -np.finfo(np.float64).epsneg]
    )

    assert_array_equal(pr.norm_angle(a_eps), a_eps)
    assert_array_equal(pr.norm_angle(a_epsneg), a_epsneg)


def test_passive_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    with pytest.raises(ValueError, match="Basis must be in"):
        pr.passive_matrix_from_angle(-1, 0)
    with pytest.raises(ValueError, match="Basis must be in"):
        pr.passive_matrix_from_angle(3, 0)

    R = pr.passive_matrix_from_angle(0, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    R = pr.passive_matrix_from_angle(0, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))

    R = pr.passive_matrix_from_angle(1, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    R = pr.passive_matrix_from_angle(1, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]))

    R = pr.passive_matrix_from_angle(2, -0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    R = pr.passive_matrix_from_angle(2, 0.5 * np.pi)
    assert_array_almost_equal(R, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))


def test_active_matrix_from_angle():
    """Sanity checks for rotation around basis vectors."""
    with pytest.raises(ValueError, match="Basis must be in"):
        pr.active_matrix_from_angle(-1, 0)
    with pytest.raises(ValueError, match="Basis must be in"):
        pr.active_matrix_from_angle(3, 0)

    rng = np.random.default_rng(21)
    for _ in range(20):
        basis = rng.integers(0, 3)
        angle = 2.0 * np.pi * rng.random() - np.pi
        R_passive = pr.passive_matrix_from_angle(basis, angle)
        R_active = pr.active_matrix_from_angle(basis, angle)
        assert_array_almost_equal(R_active, R_passive.T)


def test_active_rotation_is_default():
    """Test that rotations are active by default."""
    Rx = pr.active_matrix_from_angle(0, 0.5 * np.pi)
    ax = np.array([1, 0, 0, 0.5 * np.pi])
    qx = pr.quaternion_from_axis_angle(ax)
    assert_array_almost_equal(Rx, pr.matrix_from_axis_angle(ax))
    assert_array_almost_equal(Rx, pr.matrix_from_quaternion(qx))
    Ry = pr.active_matrix_from_angle(1, 0.5 * np.pi)
    ay = np.array([0, 1, 0, 0.5 * np.pi])
    qy = pr.quaternion_from_axis_angle(ay)
    assert_array_almost_equal(Ry, pr.matrix_from_axis_angle(ay))
    assert_array_almost_equal(Ry, pr.matrix_from_quaternion(qy))
    Rz = pr.active_matrix_from_angle(2, 0.5 * np.pi)
    az = np.array([0, 0, 1, 0.5 * np.pi])
    qz = pr.quaternion_from_axis_angle(az)
    assert_array_almost_equal(Rz, pr.matrix_from_axis_angle(az))
    assert_array_almost_equal(Rz, pr.matrix_from_quaternion(qz))


def test_quaternion_from_angle():
    """Quaternion from rotation around basis vectors."""
    with pytest.raises(ValueError, match="Basis must be in"):
        pr.quaternion_from_angle(-1, 0)
    with pytest.raises(ValueError, match="Basis must be in"):
        pr.quaternion_from_angle(3, 0)

    rng = np.random.default_rng(22)
    for _ in range(20):
        basis = rng.integers(0, 3)
        angle = 2.0 * np.pi * rng.random() - np.pi
        R = pr.active_matrix_from_angle(basis, angle)
        q = pr.quaternion_from_angle(basis, angle)
        Rq = pr.matrix_from_quaternion(q)
        assert_array_almost_equal(R, Rq)
