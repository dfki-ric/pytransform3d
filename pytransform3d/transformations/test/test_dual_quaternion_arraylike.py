"""Public scalar dual-quaternion assertion input-contract regressions."""

import numpy as np
import pytest
from pytransform3d.transformations import (
    assert_unit_dual_quaternion_equal,
    transform_from_dual_quaternion,
)


def _readonly(values):
    result = np.array(values)
    result.setflags(write=False)
    return result


CONTAINERS = [np.array, list, tuple, _readonly]
POSE = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.5])


@pytest.mark.parametrize("left", CONTAINERS)
@pytest.mark.parametrize("right", CONTAINERS)
@pytest.mark.parametrize("negate", [False, True])
def test_equivalent_array_like_dual_quaternions(left, right, negate):
    equivalent = -POSE if negate else POSE
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(
        transform_from_dual_quaternion(POSE), expected
    )
    np.testing.assert_array_equal(
        transform_from_dual_quaternion(equivalent), expected
    )
    a, b = left(POSE), right(equivalent)
    before_a, before_b = np.array(a, copy=True), np.array(b, copy=True)
    assert_unit_dual_quaternion_equal(a, b)
    np.testing.assert_array_equal(a, before_a)
    np.testing.assert_array_equal(b, before_b)


@pytest.mark.parametrize("left", CONTAINERS)
@pytest.mark.parametrize("right", CONTAINERS)
@pytest.mark.parametrize("negate", [False, True])
def test_different_translations_rejected(left, right, negate):
    other = POSE.copy()
    other[5] = 1.0
    if negate:
        other = -other
    with pytest.raises(AssertionError):
        assert_unit_dual_quaternion_equal(left(POSE), right(other))


@pytest.mark.parametrize("right", CONTAINERS)
def test_decimal_option_preserved(right):
    other = -POSE.copy()
    other[5] += 1.0e-4
    assert_unit_dual_quaternion_equal(POSE, right(other), decimal=3)
    with pytest.raises(AssertionError):
        assert_unit_dual_quaternion_equal(POSE, right(other), decimal=6)


@pytest.mark.parametrize("right", CONTAINERS)
def test_does_not_normalize_inputs(right):
    with pytest.raises(AssertionError):
        assert_unit_dual_quaternion_equal(POSE, right(-2.0 * POSE))
