import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_check_rotor():
    r_list = [1, 0, 0, 0]
    r = pr.check_rotor(r_list)
    assert isinstance(r, np.ndarray)

    r2 = np.array([[1, 0, 0, 0]])
    with pytest.raises(ValueError, match="Expected rotor with shape"):
        pr.check_rotor(r2)

    r3 = np.array([1, 0, 0])
    with pytest.raises(ValueError, match="Expected rotor with shape"):
        pr.check_rotor(r3)

    r4 = np.array([2, 0, 0, 0])
    assert pytest.approx(np.linalg.norm(pr.check_rotor(r4))) == 1.0


def test_outer():
    rng = np.random.default_rng(82)
    for _ in range(5):
        a = rng.standard_normal(3)
        Zero = pr.wedge(a, a)
        assert_array_almost_equal(Zero, np.zeros(3))

        b = rng.standard_normal(3)
        A = pr.wedge(a, b)
        B = pr.wedge(b, a)
        assert_array_almost_equal(A, -B)

        c = rng.standard_normal(3)
        assert_array_almost_equal(pr.wedge(a, (b + c)), A + pr.wedge(a, c))


def test_plane_normal_from_bivector():
    rng = np.random.default_rng(82)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        B = pr.wedge(a, b)
        n = pr.plane_normal_from_bivector(B)
        assert_array_almost_equal(n, pr.norm_vector(np.cross(a, b)))


def test_geometric_product():
    rng = np.random.default_rng(83)
    for _ in range(5):
        a = rng.standard_normal(3)
        a2a = pr.geometric_product(a, 2 * a)
        assert_array_almost_equal(a2a, np.array([np.dot(a, 2 * a), 0, 0, 0]))

        b = pr.perpendicular_to_vector(a)
        ab = pr.geometric_product(a, b)
        assert_array_almost_equal(ab, np.hstack(((0.0,), pr.wedge(a, b))))


def test_geometric_product_creates_rotor_that_rotates_by_double_angle():
    rng = np.random.default_rng(83)
    for _ in range(5):
        a_unit = pr.norm_vector(rng.standard_normal(3))
        b_unit = pr.norm_vector(rng.standard_normal(3))
        # a geometric product of two unit vectors is a rotor
        ab = pr.geometric_product(a_unit, b_unit)
        assert pytest.approx(np.linalg.norm(ab)) == 1.0

        angle = pr.angle_between_vectors(a_unit, b_unit)
        c = pr.rotor_apply(ab, a_unit)
        double_angle = pr.angle_between_vectors(a_unit, c)

        assert pytest.approx(abs(pr.norm_angle(2.0 * angle))) == abs(
            pr.norm_angle(double_angle)
        )


def test_rotor_from_two_directions_special_cases():
    d1 = np.array([0, 0, 1])
    rotor = pr.rotor_from_two_directions(d1, d1)
    assert_array_almost_equal(rotor, np.array([1, 0, 0, 0]))

    rotor = pr.rotor_from_two_directions(d1, np.zeros(3))
    assert_array_almost_equal(rotor, np.array([1, 0, 0, 0]))

    d2 = np.array([0, 0, -1])  # 180 degree rotation
    rotor = pr.rotor_from_two_directions(d1, d2)
    assert_array_almost_equal(rotor, np.array([0, 1, 0, 0]))
    assert_array_almost_equal(pr.rotor_apply(rotor, d1), d2)

    d3 = np.array([1, 0, 0])
    d4 = np.array([-1, 0, 0])
    rotor = pr.rotor_from_two_directions(d3, d4)
    assert_array_almost_equal(pr.rotor_apply(rotor, d3), d4)


def test_rotor_from_two_directions():
    rng = np.random.default_rng(84)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        rotor = pr.rotor_from_two_directions(a, b)
        b2 = pr.rotor_apply(rotor, a)
        assert_array_almost_equal(pr.norm_vector(b), pr.norm_vector(b2))
        assert_array_almost_equal(np.linalg.norm(a), np.linalg.norm(b2))


def test_rotor_concatenation():
    rng = np.random.default_rng(85)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        c = rng.standard_normal(3)
        rotor_ab = pr.rotor_from_two_directions(a, b)
        rotor_bc = pr.rotor_from_two_directions(b, c)
        rotor_ac = pr.rotor_from_two_directions(a, c)
        rotor_ac_cat = pr.concatenate_rotors(rotor_bc, rotor_ab)
        assert_array_almost_equal(
            pr.rotor_apply(rotor_ac, a), pr.rotor_apply(rotor_ac_cat, a)
        )


def test_rotor_times_reverse():
    rng = np.random.default_rng(85)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        rotor = pr.rotor_from_two_directions(a, b)
        rotor_reverse = pr.rotor_reverse(rotor)
        result = pr.concatenate_rotors(rotor, rotor_reverse)
        assert_array_almost_equal(result, [1, 0, 0, 0])


def test_rotor_from_plane_angle():
    rng = np.random.default_rng(87)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        B = pr.wedge(a, b)
        angle = 2 * np.pi * rng.random()
        axis = np.cross(a, b)
        rotor = pr.rotor_from_plane_angle(B, angle)
        q = pr.quaternion_from_axis_angle(np.r_[axis, angle])
        v = rng.standard_normal(3)
        assert_array_almost_equal(
            pr.rotor_apply(rotor, v), pr.q_prod_vector(q, v)
        )


def test_matrix_from_rotor():
    rng = np.random.default_rng(88)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        B = pr.wedge(a, b)
        angle = 2 * np.pi * rng.random()
        axis = np.cross(a, b)
        rotor = pr.rotor_from_plane_angle(B, angle)
        R_rotor = pr.matrix_from_rotor(rotor)
        q = pr.quaternion_from_axis_angle(np.r_[axis, angle])
        R_q = pr.matrix_from_quaternion(q)
        assert_array_almost_equal(R_rotor, R_q)


def test_negative_rotor():
    rng = np.random.default_rng(89)
    for _ in range(5):
        a = rng.standard_normal(3)
        b = rng.standard_normal(3)
        rotor = pr.rotor_from_two_directions(a, b)
        neg_rotor = pr.norm_vector(-rotor)
        a2 = pr.rotor_apply(rotor, a)
        a3 = pr.rotor_apply(neg_rotor, a)
        assert_array_almost_equal(a2, a3)
