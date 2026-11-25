"""Quaternion trajectory smoothing with integral storage."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytransform3d.batch_rotations import smooth_quaternion_trajectory


@pytest.mark.parametrize("dtype", [np.int8, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("container", [np.asarray, list, tuple])
@pytest.mark.parametrize("start_component_positive", ["w", "x"])
def test_smooth_signed_quaternion_trajectory(
    dtype, container, start_component_positive
):
    q = np.array(
        [-1, 0, 0, 0] if start_component_positive == "w" else [0, -1, 0, 0],
        dtype=dtype,
    )
    trajectory = np.array([q, -q, q, q, -q])
    original = trajectory.copy()
    trajectory.setflags(write=False)
    result = smooth_quaternion_trajectory(
        container(trajectory), start_component_positive
    )
    assert_allclose(result, np.tile(-q, (len(trajectory), 1)))
    assert_array_equal(trajectory, original)
    assert not np.shares_memory(result, trajectory)


@pytest.mark.parametrize(
    "dtype", [np.bool_, np.uint8, np.uint64, np.int8, np.int64]
)
@pytest.mark.parametrize("component", ["w", "x", "y", "z"])
def test_smooth_single_positive_integral_quaternion(dtype, component):
    q = np.eye(4, dtype=dtype)["wxyz".index(component)]
    trajectory = np.tile(q, (3, 1))
    assert_allclose(
        smooth_quaternion_trajectory(trajectory, component), trajectory
    )


def test_smooth_unsigned_distance_does_not_wrap():
    trajectory = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.uint8
    )
    assert_allclose(
        smooth_quaternion_trajectory(trajectory),
        smooth_quaternion_trajectory(trajectory.astype(float)),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_smooth_preserves_floating_dtype(dtype):
    trajectory = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=dtype)
    assert smooth_quaternion_trajectory(trajectory).dtype == dtype
