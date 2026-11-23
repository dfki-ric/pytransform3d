import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytransform3d.trajectories import invert_transforms


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.uint64, np.int8, np.int64, np.bool_]
)
@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_integral_inverse_translations(dtype, shape):
    transform = np.eye(4, dtype=dtype)
    transform[:3, 3] = [1, 2, 3]
    transforms = np.broadcast_to(transform, shape + (4, 4))
    before = transforms.copy()
    actual = invert_transforms(transforms)
    expected = np.linalg.inv(transforms.astype(float))
    assert_allclose(actual, expected, atol=1e-15)
    assert_allclose(
        actual @ transforms, np.broadcast_to(np.eye(4), actual.shape)
    )
    assert_array_equal(transforms, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
def test_inverse_float_precision_and_rotation(dtype, shape):
    angle = 0.4
    c, s = np.cos(angle), np.sin(angle)
    transform = np.array(
        [[c, -s, 0, 1], [s, c, 0, 2], [0, 0, 1, -3], [0, 0, 0, 1]], dtype=dtype
    )
    transforms = np.broadcast_to(transform, shape + (4, 4))
    actual = invert_transforms(transforms)
    assert actual.dtype == dtype
    assert_allclose(actual, np.linalg.inv(transforms), atol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint8, np.bool_, np.float32])
def test_empty_inverse_batch(dtype):
    values = np.empty((0, 4, 4), dtype=dtype)
    assert invert_transforms(values).shape == values.shape
