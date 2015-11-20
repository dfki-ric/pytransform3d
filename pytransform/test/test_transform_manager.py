import numpy as np
from pytransform.transformations import (invert_transform, translate_transform,
                                         concat)
from pytransform.transform_manager import TransformManager
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises_regexp


def test_request_added_transform():
    """Request an added transform from the transform manager."""
    A2B = np.eye(4)

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    A2B_2 = tm.get_transform("A", "B")
    assert_array_almost_equal(A2B, A2B_2)


def test_request_inverse_transform():
    """Request an inverse transform from the transform manager."""
    A2B = np.eye(4)
    translate_transform(A2B, np.array([0.3, 0.5, -0.1]))

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    A2B_2 = tm.get_transform("A", "B")
    assert_array_almost_equal(A2B, A2B_2)

    B2A = tm.get_transform("B", "A")
    B2A_2 = invert_transform(A2B)
    assert_array_almost_equal(B2A, B2A_2)


def test_request_concatenated_transform():
    """Request a concatenated transform from the transform manager."""
    # TODO make more random transforms
    A2B = np.eye(4)
    translate_transform(A2B, np.array([0.3, 0.5, -0.1]))
    B2C = np.eye(4)
    translate_transform(B2C, np.array([0.1, 0.9, -0.8]))
    A2F = np.eye(4)
    translate_transform(A2F, np.array([0.1, -0.9, 0.8]))

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    tm.add_transform("B", "C", B2C)
    tm.add_transform("D", "E", np.eye(4))
    tm.add_transform("A", "F", A2F)

    A2C = tm.get_transform("A", "C")
    assert_array_almost_equal(A2C, concat(A2B, B2C))

    C2A = tm.get_transform("C", "A")
    assert_array_almost_equal(C2A, concat(invert_transform(B2C),
                                          invert_transform(A2B)))

    F2B = tm.get_transform("F", "B")
    assert_array_almost_equal(F2B, concat(invert_transform(A2F), A2B))

    assert_raises_regexp(KeyError, "Unknown frame", tm.get_transform, "A", "G")

    assert_raises_regexp(KeyError, "Unknown frame", tm.get_transform, "G", "D")

    assert_raises_regexp(KeyError, "Cannot compute path", tm.get_transform,
                         "A", "D")
