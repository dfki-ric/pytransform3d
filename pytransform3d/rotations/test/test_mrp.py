import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_check_mrp():
    with pytest.raises(
        ValueError, match="Expected modified Rodrigues parameters with shape"
    ):
        pr.check_mrp([])
    with pytest.raises(
        ValueError, match="Expected modified Rodrigues parameters with shape"
    ):
        pr.check_mrp(np.zeros((3, 4)))


def test_norm_mrp():
    mrp_norm = pr.norm_mrp(pr.mrp_from_axis_angle([1.0, 0.0, 0.0, 1.5 * np.pi]))
    assert_array_almost_equal(
        [-1.0, 0.0, 0.0, 0.5 * np.pi], pr.axis_angle_from_mrp(mrp_norm)
    )

    mrp_norm = pr.norm_mrp(
        pr.mrp_from_axis_angle([1.0, 0.0, 0.0, -0.5 * np.pi])
    )
    assert_array_almost_equal(
        [-1.0, 0.0, 0.0, 0.5 * np.pi], pr.axis_angle_from_mrp(mrp_norm)
    )

    mrp_norm = pr.norm_mrp(pr.mrp_from_axis_angle([1.0, 0.0, 0.0, 2.0 * np.pi]))
    assert_array_almost_equal(
        [1.0, 0.0, 0.0, 0.0], pr.axis_angle_from_mrp(mrp_norm)
    )

    mrp_norm = pr.norm_mrp(
        pr.mrp_from_axis_angle([1.0, 0.0, 0.0, -2.0 * np.pi])
    )
    assert_array_almost_equal(
        [1.0, 0.0, 0.0, 0.0], pr.axis_angle_from_mrp(mrp_norm)
    )


def test_mrp_near_singularity():
    axis = np.array([1.0, 0.0, 0.0])
    assert pr.mrp_near_singularity(np.tan(2.0 * np.pi / 4.0) * axis)
    assert pr.mrp_near_singularity(np.tan(2.0 * np.pi / 4.0 - 1e-7) * axis)
    assert pr.mrp_near_singularity(np.tan(2.0 * np.pi / 4.0 + 1e-7) * axis)
    assert not pr.mrp_near_singularity(np.tan(np.pi / 4.0) * axis)
    assert not pr.mrp_near_singularity(np.tan(0.0 / 4.0) * axis)


def test_mrp_double():
    rng = np.random.default_rng(23238)
    mrp = pr.random_vector(rng, 3)
    mrp_double = pr.mrp_double(mrp)
    q = pr.quaternion_from_mrp(mrp)
    q_double = pr.quaternion_from_mrp(mrp_double)
    pr.assert_mrp_equal(mrp, mrp_double)
    assert not np.allclose(mrp, mrp_double)
    pr.assert_quaternion_equal(q, q_double)
    assert not np.allclose(q, q_double)

    assert_array_almost_equal(np.zeros(3), pr.mrp_double(np.zeros(3)))


def test_concatenate_mrp():
    rng = np.random.default_rng(283)
    for _ in range(5):
        q1 = pr.random_quaternion(rng)
        q2 = pr.random_quaternion(rng)
        q12 = pr.concatenate_quaternions(q1, q2)
        mrp1 = pr.mrp_from_quaternion(q1)
        mrp2 = pr.mrp_from_quaternion(q2)
        mrp12 = pr.concatenate_mrp(mrp1, mrp2)
        pr.assert_quaternion_equal(q12, pr.quaternion_from_mrp(mrp12))


def test_mrp_prod_vector():
    rng = np.random.default_rng(2183)
    v = pr.random_vector(rng, 3)
    assert_array_almost_equal(v, pr.mrp_prod_vector([0, 0, 0], v))

    for _ in range(5):
        mrp = pr.random_vector(rng, 3)
        q = pr.quaternion_from_mrp(mrp)
        v_mrp = pr.mrp_prod_vector(mrp, v)
        v_q = pr.q_prod_vector(q, v)
        assert_array_almost_equal(v_mrp, v_q)


@pytest.mark.parametrize(
    "mrp, v", [([1, 0, 0], [1, 0, 0]), ([0, 0.5, 0], [0, 2, 0])]
)
def test_mrp_prod_vector_along_axis(mrp, v):
    assert_array_almost_equal(v, pr.mrp_prod_vector(mrp, v))


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.float32, np.float64])
def test_mrp_prod_vector_integer_mrp(dtype):
    mrp = np.array([12, 0, 0], dtype=dtype)
    v = [0.0, 1.0, 0.0]
    assert_array_almost_equal(
        pr.mrp_prod_vector(mrp.astype(float), v), pr.mrp_prod_vector(mrp, v)
    )


@pytest.mark.parametrize("scale", [0.0, 1e-8, 0.5, 1.0, 2.0, 1e8, 1e100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mrp_prod_vector_direct_formula(scale, dtype):
    # Large MRPs approach the 2*pi singularity; no denominator square is needed.
    mrp = scale * np.array([0.0, 0.6, 0.8])
    v = np.array([1.0, -2.0, 3.0], dtype=dtype)
    mrp_before, v_before = mrp.copy(), v.copy()
    np.testing.assert_array_equal(pr.mrp_prod_vector(mrp, [0, 0, 0]), [0, 0, 0])
    rotated = pr.mrp_prod_vector(mrp, v)
    expected = pr.q_prod_vector(pr.quaternion_from_mrp(mrp), v)
    np.testing.assert_allclose(rotated, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(rotated), np.linalg.norm(v.astype(float))
    )
    np.testing.assert_allclose(pr.mrp_prod_vector(-mrp, rotated), v, atol=1e-12)
    np.testing.assert_allclose(
        pr.mrp_prod_vector(mrp, 2.0 * v), 2.0 * rotated, atol=1e-12
    )
    if scale != 0.0:
        np.testing.assert_allclose(
            pr.mrp_prod_vector(pr.mrp_double(mrp), v), rotated, atol=1e-12
        )
    np.testing.assert_array_equal(mrp, mrp_before)
    np.testing.assert_array_equal(v, v_before)


@pytest.mark.parametrize("offset", [-1e-8, 0.0, 1e-8])
def test_mrp_prod_vector_near_singular_concatenation(offset):
    mrp = [1.0, 0.0, 0.0]
    v = [1.0 + offset, offset, 0.0]
    np.testing.assert_allclose(
        pr.mrp_prod_vector(mrp, v), [1.0 + offset, -offset, 0.0], atol=1e-15
    )


@pytest.mark.parametrize(
    "mrp, v",
    [
        ([], [1, 0, 0]),
        ([0, 0, 0], []),
        ([[1, 0, 0]], [1, 0, 0]),
        ([1, 0, 0], [[1, 0, 0]]),
    ],
)
def test_mrp_prod_vector_invalid_shape(mrp, v):
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        pr.mrp_prod_vector(mrp, v)


def test_mrp_quat_conversions():
    rng = np.random.default_rng(22)

    for _ in range(5):
        q = pr.random_quaternion(rng)
        mrp = pr.mrp_from_quaternion(q)
        q2 = pr.quaternion_from_mrp(mrp)
        pr.assert_quaternion_equal(q, q2)


def test_axis_angle_from_mrp():
    rng = np.random.default_rng(98343)
    for _ in range(5):
        mrp = pr.random_vector(rng, 3)
        a = pr.axis_angle_from_mrp(mrp)
        q = pr.quaternion_from_mrp(mrp)
        pr.assert_axis_angle_equal(a, pr.axis_angle_from_quaternion(q))

    pr.assert_axis_angle_equal(
        pr.axis_angle_from_mrp([np.tan(0.5 * np.pi), 0.0, 0.0]),
        [1.0, 0.0, 0.0, 0.0],
    )

    pr.assert_axis_angle_equal(
        pr.axis_angle_from_mrp([0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]
    )
