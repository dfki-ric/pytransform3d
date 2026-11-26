"""Array-like endpoints in batch quaternion interpolation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytransform3d.batch_rotations import quaternion_slerp_batch
from pytransform3d.rotations import quaternion_slerp


@pytest.mark.parametrize("start_type", [np.asarray, list, tuple])
@pytest.mark.parametrize("end_type", [np.asarray, list, tuple])
@pytest.mark.parametrize("shortest_path", [False, True])
@pytest.mark.parametrize("negative_end", [False, True])
def test_slerp_array_like_endpoints(
    start_type, end_type, shortest_path, negative_end
):
    start = np.array([1.0, 0, 0, 0])
    end = np.array([0.5, 0.5, 0.5, 0.5]) * (-1 if negative_end else 1)
    original_start, original_end = start.copy(), end.copy()
    start.setflags(write=False)
    end.setflags(write=False)
    ts = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = np.array(
        [quaternion_slerp(start, end, t, shortest_path) for t in ts]
    )
    actual = quaternion_slerp_batch(
        start_type(start), end_type(end), ts, shortest_path
    )
    assert_allclose(actual, expected, atol=1e-15)
    assert_array_equal(start, original_start)
    assert_array_equal(end, original_end)


@pytest.mark.parametrize("container", [np.asarray, list, tuple])
def test_slerp_array_like_empty_steps(container):
    assert quaternion_slerp_batch(
        container([1, 0, 0, 0]), container([0, 1, 0, 0]), []
    ).shape == (0, 4)


@pytest.mark.parametrize("container", [np.asarray, list, tuple])
def test_slerp_array_like_identical_endpoints(container):
    q = container([1, 0, 0, 0])
    assert_allclose(
        quaternion_slerp_batch(q, q, [0, 0.5, 1]), [[1, 0, 0, 0]] * 3
    )
