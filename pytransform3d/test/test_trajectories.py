import numpy as np
from pytransform3d.trajectories import transforms_from_pqs, pqs_from_transforms
from pytransform3d.rotations import (random_quaternion, quaternion_from_matrix,
                                     assert_quaternion_equal)
from pytransform3d.batch_rotations import norm_vectors
from numpy.testing import assert_array_almost_equal


def test_matrices_from_pos_quat():
    """Test conversion from positions and quaternions to matrices."""
    P = np.empty((10, 7))
    random_state = np.random.RandomState(0)
    P[:, :3] = random_state.randn(len(P), 3)
    P[:, 3:] = norm_vectors(random_state.randn(len(P), 4))

    H = transforms_from_pqs(P)
    P2 = pqs_from_transforms(H)

    assert_array_almost_equal(P[:, :3], H[:, :3, 3])
    assert_array_almost_equal(P[:, :3], P2[:, :3])

    for t in range(len(P)):
        assert_quaternion_equal(P[t, 3:], quaternion_from_matrix(H[t, :3, :3]))
        assert_quaternion_equal(P[t, 3:], P2[t, 3:])
