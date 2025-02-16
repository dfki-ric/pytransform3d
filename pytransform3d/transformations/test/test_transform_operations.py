import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def test_invert_transform():
    """Test inversion of transformations."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        R = pr.matrix_from_axis_angle(pr.random_axis_angle(rng))
        p = pr.random_vector(rng)
        A2B = pt.transform_from(R, p)
        B2A = pt.invert_transform(A2B)
        A2B2 = np.linalg.inv(B2A)
        assert_array_almost_equal(A2B, A2B2)


def test_invert_transform_without_check():
    """Test inversion of transformations."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        A2B = rng.standard_normal(size=(4, 4))
        A2B = A2B + A2B.T
        B2A = pt.invert_transform(A2B, check=False)
        A2B2 = np.linalg.inv(B2A)
        assert_array_almost_equal(A2B, A2B2)


def test_vector_to_point():
    """Test conversion from vector to homogenous coordinates."""
    v = np.array([1, 2, 3])
    pA = pt.vector_to_point(v)
    assert_array_almost_equal(pA, [1, 2, 3, 1])

    rng = np.random.default_rng(0)
    R = pr.matrix_from_axis_angle(pr.random_axis_angle(rng))
    p = pr.random_vector(rng)
    A2B = pt.transform_from(R, p)
    pt.assert_transform(A2B)
    _ = pt.transform(A2B, pA)


def test_vectors_to_points():
    """Test conversion from vectors to homogenous coordinates."""
    V = np.array([[1, 2, 3], [2, 3, 4]])
    PA = pt.vectors_to_points(V)
    assert_array_almost_equal(PA, [[1, 2, 3, 1], [2, 3, 4, 1]])

    rng = np.random.default_rng(0)
    V = rng.standard_normal(size=(10, 3))
    for i, p in enumerate(pt.vectors_to_points(V)):
        assert_array_almost_equal(p, pt.vector_to_point(V[i]))


def test_vector_to_direction():
    """Test conversion from vector to direction in homogenous coordinates."""
    v = np.array([1, 2, 3])
    dA = pt.vector_to_direction(v)
    assert_array_almost_equal(dA, [1, 2, 3, 0])

    rng = np.random.default_rng(0)
    R = pr.matrix_from_axis_angle(pr.random_axis_angle(rng))
    p = pr.random_vector(rng)
    A2B = pt.transform_from(R, p)
    pt.assert_transform(A2B)
    _ = pt.transform(A2B, dA)


def test_vectors_to_directions():
    """Test conversion from vectors to directions in homogenous coordinates."""
    V = np.array([[1, 2, 3], [2, 3, 4]])
    DA = pt.vectors_to_directions(V)
    assert_array_almost_equal(DA, [[1, 2, 3, 0], [2, 3, 4, 0]])

    rng = np.random.default_rng(0)
    V = rng.standard_normal(size=(10, 3))
    for i, d in enumerate(pt.vectors_to_directions(V)):
        assert_array_almost_equal(d, pt.vector_to_direction(V[i]))


def test_concat():
    """Test concatenation of transforms."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        A2B = pt.random_transform(rng)
        pt.assert_transform(A2B)

        B2C = pt.random_transform(rng)
        pt.assert_transform(B2C)

        A2C = pt.concat(A2B, B2C)
        pt.assert_transform(A2C)

        p_A = np.array([0.3, -0.2, 0.9, 1.0])
        p_C = pt.transform(A2C, p_A)

        C2A = pt.invert_transform(A2C)
        p_A2 = pt.transform(C2A, p_C)

        assert_array_almost_equal(p_A, p_A2)

        C2A2 = pt.concat(pt.invert_transform(B2C), pt.invert_transform(A2B))
        p_A3 = pt.transform(C2A2, p_C)
        assert_array_almost_equal(p_A, p_A3)


def test_transform():
    """Test transformation of points."""
    PA = np.array([[1, 2, 3, 1], [2, 3, 4, 1]])

    rng = np.random.default_rng(0)
    A2B = pt.random_transform(rng)

    PB = pt.transform(A2B, PA)
    p0B = pt.transform(A2B, PA[0])
    p1B = pt.transform(A2B, PA[1])
    assert_array_almost_equal(PB, np.array([p0B, p1B]))

    with pytest.raises(
        ValueError, match="Cannot transform array with more than 2 dimensions"
    ):
        pt.transform(A2B, np.zeros((2, 2, 4)))


def test_scale_transform():
    """Test scaling of transforms."""
    rng = np.random.default_rng(0)
    A2B = pt.random_transform(rng)

    A2B_scaled1 = pt.scale_transform(A2B, s_xt=0.5, s_yt=0.5, s_zt=0.5)
    A2B_scaled2 = pt.scale_transform(A2B, s_t=0.5)
    assert_array_almost_equal(A2B_scaled1, A2B_scaled2)

    A2B_scaled1 = pt.scale_transform(A2B, s_xt=0.5, s_yt=0.5, s_zt=0.5, s_r=0.5)
    A2B_scaled2 = pt.scale_transform(A2B, s_t=0.5, s_r=0.5)
    A2B_scaled3 = pt.scale_transform(A2B, s_d=0.5)
    assert_array_almost_equal(A2B_scaled1, A2B_scaled2)
    assert_array_almost_equal(A2B_scaled1, A2B_scaled3)

    A2B_scaled = pt.scale_transform(A2B, s_xr=0.0)
    a_scaled = pr.axis_angle_from_matrix(A2B_scaled[:3, :3])
    assert_array_almost_equal(a_scaled[0], 0.0)
    A2B_scaled = pt.scale_transform(A2B, s_yr=0.0)
    a_scaled = pr.axis_angle_from_matrix(A2B_scaled[:3, :3])
    assert_array_almost_equal(a_scaled[1], 0.0)
    A2B_scaled = pt.scale_transform(A2B, s_zr=0.0)
    a_scaled = pr.axis_angle_from_matrix(A2B_scaled[:3, :3])
    assert_array_almost_equal(a_scaled[2], 0.0)
