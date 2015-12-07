import numpy as np
from pytransform.trajectories import matrices_from_pos_quat
from pytransform.rotations import (random_quaternion, quaternion_from_matrix,
                                   assert_quaternion_equal)
from numpy.testing import assert_array_almost_equal


def test_matrices_from_pos_quat():
    """Test conversion from positions and quaternions to matrices."""
    P = np.empty((10, 7))
    random_state = np.random.RandomState(0)
    P[:, :3] = random_state.randn(10, 3)
    for t in range(10):
        P[t, 3:] = random_quaternion(random_state)

    H = matrices_from_pos_quat(P)
    for t in range(10):
        assert_array_almost_equal(P[t, :3], H[t, :3, 3])
        assert_quaternion_equal(P[t, 3:], quaternion_from_matrix(H[t, :3, :3]))
