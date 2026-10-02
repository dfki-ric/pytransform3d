import warnings

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)

import pytransform3d.rotations as pr


def test_norm_vector():
    """Test normalization of vectors."""
    rng = np.random.default_rng(0)
    for n in range(1, 6):
        v = pr.random_vector(rng, n)
        u = pr.norm_vector(v)
        assert pytest.approx(np.linalg.norm(u)) == 1


def test_norm_zero_vector():
    """Test normalization of zero vector."""
    normalized = pr.norm_vector(np.zeros(3))
    assert np.isfinite(np.linalg.norm(normalized))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "magnitude", ["subnormal", "tiny", "small", "ordinary", "large", "maximum"]
)
def test_norm_vector_extreme_magnitudes(dtype, magnitude):
    finfo = np.finfo(dtype)
    scale = {
        "subnormal": finfo.smallest_subnormal,
        "tiny": finfo.tiny / dtype(8),
        "small": np.sqrt(finfo.tiny) / dtype(10),
        "ordinary": dtype(1),
        "large": np.sqrt(finfo.max) * dtype(2),
        "maximum": finfo.max / dtype(4),
    }[magnitude]
    v = np.array([4, -2, 1], dtype=dtype) * scale
    original = v.copy()
    expected = np.array([4.0, -2.0, 1.0]) / np.sqrt(21.0)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized = pr.norm_vector(v)

    assert normalized.dtype == dtype
    assert_allclose(normalized, expected, rtol=4 * finfo.eps, atol=0)
    assert_allclose(np.linalg.norm(normalized), 1, rtol=4 * finfo.eps)
    assert_array_equal(v, original)


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_norm_vector_extreme_lists(scale):
    normalized = pr.norm_vector([4 * scale, -2 * scale, scale])
    assert_allclose(normalized, np.array([4, -2, 1]) / np.sqrt(21.0))


@pytest.mark.parametrize("v", [[], [0, 0, 0], np.zeros(3)])
def test_norm_vector_keeps_zero_input(v):
    assert pr.norm_vector(v) is v


@pytest.mark.parametrize(
    "v, expected",
    [
        ([3, 4, 0], [0.6, 0.8, 0]),
        (np.array([np.iinfo(np.int64).min, 0, 0]), [-1, 0, 0]),
        (np.array([True, True, False]), [1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
    ],
)
def test_norm_vector_integer_inputs(v, expected):
    assert_allclose(pr.norm_vector(v), expected)


def test_norm_vector_complex_input():
    v = np.array([3j, 4])
    assert_allclose(pr.norm_vector(v), [0.6j, 0.8])


@pytest.mark.parametrize("v", [[np.inf, 1], [-np.inf, 1], [np.nan, 1]])
def test_norm_vector_nonfinite_input(v):
    with np.errstate(invalid="ignore"):
        expected = np.asarray(v) / np.linalg.norm(v)
        actual = pr.norm_vector(v)
    assert_array_equal(actual, expected)


def test_perpendicular_to_vectors():
    """Test function to compute perpendicular to vectors."""
    rng = np.random.default_rng(0)
    a = pr.norm_vector(pr.random_vector(rng))
    a1 = pr.norm_vector(pr.random_vector(rng))
    b = pr.norm_vector(pr.perpendicular_to_vectors(a, a1))
    c = pr.norm_vector(pr.perpendicular_to_vectors(a, b))
    assert pytest.approx(pr.angle_between_vectors(a, b)) == np.pi / 2.0
    assert pytest.approx(pr.angle_between_vectors(a, c)) == np.pi / 2.0
    assert pytest.approx(pr.angle_between_vectors(b, c)) == np.pi / 2.0
    assert_array_almost_equal(pr.perpendicular_to_vectors(b, c), a)
    assert_array_almost_equal(pr.perpendicular_to_vectors(c, a), b)


def test_perpendicular_to_vector():
    """Test function to compute perpendicular to vector."""
    assert (
        pytest.approx(
            pr.angle_between_vectors(
                pr.unitx, pr.perpendicular_to_vector(pr.unitx)
            )
        )
        == np.pi / 2.0
    )
    assert (
        pytest.approx(
            pr.angle_between_vectors(
                pr.unity, pr.perpendicular_to_vector(pr.unity)
            )
        )
        == np.pi / 2.0
    )
    assert (
        pytest.approx(
            pr.angle_between_vectors(
                pr.unitz, pr.perpendicular_to_vector(pr.unitz)
            )
        )
        == np.pi / 2.0
    )
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = pr.norm_vector(pr.random_vector(rng))
        assert (
            pytest.approx(
                pr.angle_between_vectors(a, pr.perpendicular_to_vector(a))
            )
            == np.pi / 2.0
        )
        b = a - np.array([a[0], 0.0, 0.0])
        assert (
            pytest.approx(
                pr.angle_between_vectors(b, pr.perpendicular_to_vector(b))
            )
            == np.pi / 2.0
        )
        c = a - np.array([0.0, a[1], 0.0])
        assert (
            pytest.approx(
                pr.angle_between_vectors(c, pr.perpendicular_to_vector(c))
            )
            == np.pi / 2.0
        )
        d = a - np.array([0.0, 0.0, a[2]])
        assert (
            pytest.approx(
                pr.angle_between_vectors(d, pr.perpendicular_to_vector(d))
            )
            == np.pi / 2.0
        )


def test_angle_between_vectors():
    """Test function to compute angle between two vectors."""
    v = np.array([1, 0, 0])
    a = np.array([0, 1, 0, np.pi / 2])
    R = pr.matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert pytest.approx(pr.angle_between_vectors(vR, v)) == a[-1]
    assert pytest.approx(pr.angle_between_vectors(vR, v, fast=True)) == a[-1]
    v = np.array([0, 1, 0])
    a = np.array([1, 0, 0, np.pi / 2])
    R = pr.matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert pytest.approx(pr.angle_between_vectors(vR, v)) == a[-1]
    v = np.array([0, 0, 1])
    a = np.array([1, 0, 0, np.pi / 2])
    R = pr.matrix_from_axis_angle(a)
    vR = np.dot(R, v)
    assert pytest.approx(pr.angle_between_vectors(vR, v)) == a[-1]


def test_angle_between_close_vectors():
    """Test angle between close vectors.

    See issue #47.
    """
    a = np.array([0.9689124217106448, 0.24740395925452294, 0.0, 0.0])
    b = np.array([0.9689124217106448, 0.247403959254523, 0.0, 0.0])
    angle = pr.angle_between_vectors(a, b)
    assert pytest.approx(angle) == 0.0


@pytest.mark.parametrize(
    "angle",
    [
        1e-4,
        1e-6,
        1e-8,
        1e-10,
    ],
)
def test_angle_between_vectors_small_angle(angle):
    # 3D
    a = np.array([1.0, 0.0, 0.0])
    b = pr.matrix_from_axis_angle([0.0, 0.0, 1.0, angle]).dot(a)
    assert pytest.approx(pr.angle_between_vectors(a, b), rel=1e-12) == angle

    # 4D, e.g., quaternions
    q1 = pr.q_id
    q2 = pr.quaternion_from_axis_angle([1.0, 0.0, 0.0, 2.0 * angle])
    assert pytest.approx(pr.angle_between_vectors(q1, q2), rel=1e-12) == angle


def test_angle_to_zero_vector_is_nan():
    """Test angle to zero vector."""
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    with warnings.catch_warnings(record=True) as w:
        angle = pr.angle_between_vectors(a, b)
        assert len(w) == 1
    assert np.isnan(angle)


def test_vector_projection_on_zero_vector():
    """Test projection on zero vector."""
    rng = np.random.default_rng(23)
    for _ in range(5):
        a = pr.random_vector(rng, 3)
        a_on_b = pr.vector_projection(a, np.zeros(3))
        assert_array_almost_equal(a_on_b, np.zeros(3))


def test_vector_projection():
    """Test orthogonal projection of one vector to another vector."""
    a = np.ones(3)
    a_on_unitx = pr.vector_projection(a, pr.unitx)
    assert_array_almost_equal(a_on_unitx, pr.unitx)
    assert pytest.approx(pr.angle_between_vectors(a_on_unitx, pr.unitx)) == 0.0

    a2_on_unitx = pr.vector_projection(2 * a, pr.unitx)
    assert_array_almost_equal(a2_on_unitx, 2 * pr.unitx)
    assert pytest.approx(pr.angle_between_vectors(a2_on_unitx, pr.unitx)) == 0.0

    a_on_unity = pr.vector_projection(a, pr.unity)
    assert_array_almost_equal(a_on_unity, pr.unity)
    assert pytest.approx(pr.angle_between_vectors(a_on_unity, pr.unity)) == 0.0

    minus_a_on_unity = pr.vector_projection(-a, pr.unity)
    assert_array_almost_equal(minus_a_on_unity, -pr.unity)
    assert (
        pytest.approx(pr.angle_between_vectors(minus_a_on_unity, pr.unity))
        == np.pi
    )

    a_on_unitz = pr.vector_projection(a, pr.unitz)
    assert_array_almost_equal(a_on_unitz, pr.unitz)
    assert pytest.approx(pr.angle_between_vectors(a_on_unitz, pr.unitz)) == 0.0

    unitz_on_a = pr.vector_projection(pr.unitz, a)
    assert_array_almost_equal(unitz_on_a, np.ones(3) / 3.0)
    assert pytest.approx(pr.angle_between_vectors(unitz_on_a, a)) == 0.0

    unitx_on_unitx = pr.vector_projection(pr.unitx, pr.unitx)
    assert_array_almost_equal(unitx_on_unitx, pr.unitx)
    assert (
        pytest.approx(pr.angle_between_vectors(unitx_on_unitx, pr.unitx)) == 0.0
    )


def test_plane_basis_from_normal():
    x, y = pr.plane_basis_from_normal(pr.unitx)
    R = np.column_stack((x, y, pr.unitx))
    pr.assert_rotation_matrix(R)

    x, y = pr.plane_basis_from_normal(pr.unity)
    R = np.column_stack((x, y, pr.unity))
    pr.assert_rotation_matrix(R)

    x, y = pr.plane_basis_from_normal(pr.unitz)
    R = np.column_stack((x, y, pr.unitz))
    pr.assert_rotation_matrix(R)

    rng = np.random.default_rng(25)
    for _ in range(5):
        normal = pr.norm_vector(rng.standard_normal(3))
        x, y = pr.plane_basis_from_normal(normal)
        R = np.column_stack((x, y, normal))
        pr.assert_rotation_matrix(R)
