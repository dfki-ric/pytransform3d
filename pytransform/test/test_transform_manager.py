import numpy as np
from pytransform.transformations import invert_transform, translate_transform
from pytransform.transform_manager import TransformManager
from numpy.testing import assert_array_almost_equal


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
