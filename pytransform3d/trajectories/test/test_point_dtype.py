"""Fractional points must survive integral dual-quaternion transforms."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytransform3d.trajectories import batch_dq_prod_vector
from pytransform3d.transformations import dq_prod_vector


@pytest.mark.parametrize(
    "dtype", [np.int8, np.int64, np.uint8, np.bool_, np.float32, np.float64]
)
@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("container", [np.asarray, list, tuple])
def test_fractional_points_under_identity(dtype, shape, container):
    dqs = np.broadcast_to(
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=dtype), shape + (8,)
    ).copy()
    points = np.broadcast_to([0.25, -1.5, 2.75], shape + (3,)).copy()
    original = dqs.copy()
    dqs.setflags(write=False)
    points.setflags(write=False)
    actual = batch_dq_prod_vector(container(dqs), points)
    assert_allclose(actual, points)
    assert_array_equal(dqs, original)


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (2,), (2, 3)])
def test_fractional_points_under_integer_pose(dtype, shape):
    # Half-turn around x and a translation of two units along x.
    dq = np.array([0, 1, 0, 0, -1, 0, 0, 0], dtype=dtype)
    point = np.array([0.25, -1.5, 2.75])
    expected = dq_prod_vector(dq.astype(float), point)
    dqs = np.broadcast_to(dq, shape + (8,))
    points = np.broadcast_to(point, shape + (3,))
    assert_allclose(
        batch_dq_prod_vector(dqs, points),
        np.broadcast_to(expected, shape + (3,)),
    )


def test_empty_points():
    assert batch_dq_prod_vector(
        np.empty((0, 8), dtype=int), np.empty((0, 3))
    ).shape == (0, 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_point_dtype_preserved(dtype):
    result = batch_dq_prod_vector(
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=dtype),
        np.array([0.25, -1.5, 2.75], dtype=dtype),
    )
    assert result.dtype == dtype


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.bool_])
def test_integral_points(dtype):
    dq = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)
    point = np.array([1, 0, 1], dtype=dtype)
    assert_allclose(batch_dq_prod_vector(dq, point), point)
