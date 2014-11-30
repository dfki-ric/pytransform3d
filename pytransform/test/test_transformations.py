import numpy as np
from pytransform.transformations import transform_from, invert_transform
from pytransform.rotations import matrix_from, random_axis_angle, random_vector
from numpy.testing import assert_array_almost_equal


def test_invert_transform():
    """Test inversion of transformations."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        R = matrix_from(a=random_axis_angle(random_state))
        p = random_vector(random_state)
        A2B = transform_from(R, p)
        B2A = invert_transform(A2B)
        A2B2 = np.linalg.inv(B2A)
        assert_array_almost_equal(A2B, A2B2)
