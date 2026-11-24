import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytransform3d.batch_rotations import cross_product_matrices


@pytest.mark.parametrize(
    "dtype", [np.uint8, np.uint16, np.uint64, np.int8, np.int64, bool]
)
@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_integral_cross_products(dtype, shape):
    vector = np.array([1, 2, 3], dtype=dtype)
    vectors = np.broadcast_to(vector, shape + (3,)).copy()
    vectors.flags.writeable = False
    original = vectors.copy()
    result = cross_product_matrices(vectors)
    x, y, z = vector.astype(float)
    expected = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    assert_allclose(result, np.broadcast_to(expected, shape + (3, 3)))
    assert_allclose(result + result.swapaxes(-1, -2), 0)
    w = np.array([0.25, -0.5, 1.5])
    assert_allclose(result @ w, np.cross(vectors.astype(float), w))
    assert_array_equal(vectors, original)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64])
def test_minimum_signed_cross_product(dtype):
    vector = np.array([np.iinfo(dtype).min, 1, 2], dtype=dtype)
    result = cross_product_matrices(vector)
    assert_allclose(result[1, 2], -float(vector[0]))
    assert_allclose(result[1, 2], -result[2, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_cross_product_control(dtype):
    vector = np.array([0.5, -1.25, 3.5], dtype=dtype)
    result = cross_product_matrices(vector)
    assert_allclose(result @ [2, 3, 4], np.cross(vector, [2, 3, 4]))


def test_array_like_and_empty_cross_products():
    expected = [[0, -3, 2], [3, 0, -1], [-2, 1, 0]]
    assert_allclose(cross_product_matrices([1, 2, 3]), expected)
    assert_allclose(cross_product_matrices((1, 2, 3)), expected)
    assert cross_product_matrices(
        np.empty((2, 0, 3), dtype=np.uint8)
    ).shape == (2, 0, 3, 3)
