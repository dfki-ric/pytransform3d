"""Sensor projection must not multiply coordinates as narrow integers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytransform3d.camera import sensor2img


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.uint8, np.int16, np.uint16, np.int64, np.float32, np.float64],
)
@pytest.mark.parametrize("container", [np.asarray, list, tuple])
@pytest.mark.parametrize("center", [None, [0.5, 1.25]])
def test_sensor_projection_does_not_overflow(dtype, container, center):
    points = np.array([[2, 3], [4, 1]], dtype=dtype)
    image_size = np.array([100, 120], dtype=dtype)
    sensor_size = np.array([8, 6], dtype=dtype)
    original = points.copy()
    points.setflags(write=False)
    expected = np.array([[25.0, 60.0], [50.0, 20.0]])
    expected += [50, 60] if center is None else center
    actual = sensor2img(container(points), sensor_size, image_size, center)
    assert_allclose(actual, expected)
    assert_array_equal(points, original)


def test_sensor_projection_empty():
    assert sensor2img(
        np.empty((0, 2), dtype=np.int8), [8, 6], [100, 120]
    ).shape == (0, 2)


def test_sensor_projection_boolean_points():
    assert_allclose(
        sensor2img([[True, False]], [2, 2], [100, 120]), [[100, 60]]
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sensor_projection_retains_float_dtype(dtype):
    actual = sensor2img(
        np.array([[2, 3]], dtype=dtype),
        np.array([8, 6], dtype=dtype),
        np.array([100, 120], dtype=dtype),
    )
    assert actual.dtype == dtype
    assert_allclose(actual, [[75, 120]])
