import numpy as np
from .rotations import matrix_from_quaternion


def matrices_from_pos_quat(P):
    """Get sequence of homogeneous matrices from positions and quaternions.

    Parameters
    ----------
    P : array-like, shape (n_steps, 7)
        Sequence of poses represented by positions and quaternions in the
        order (x, y, z, w, vx, vy, vz) for each step

    Returns
    -------
    H : array-like, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices
    """
    n_steps = len(P)
    H = np.empty((n_steps, 4, 4))
    H[:, :3, 3] = P[:, :3]
    H[:, 3, :3] = 0.0
    H[:, 3, 3] = 1.0
    for t in range(n_steps):
        H[t, :3, :3] = matrix_from_quaternion(P[t, 3:])
    return H
