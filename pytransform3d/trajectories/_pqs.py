import numpy as np

from ..batch_rotations import (
    matrices_from_quaternions,
    batch_concatenate_quaternions,
)


def transforms_from_pqs(P, normalize_quaternions=True):
    """Get sequence of homogeneous matrices from positions and quaternions.

    Parameters
    ----------
    P : array-like, shape (..., 7)
        Poses represented by positions and quaternions in the
        order (x, y, z, qw, qx, qy, qz)

    normalize_quaternions : bool, optional (default: True)
        Normalize quaternions before conversion

    Returns
    -------
    A2Bs : array, shape (..., 4, 4)
        Poses represented by homogeneous matrices
    """
    P = np.asarray(P)
    instances_shape = P.shape[:-1]
    A2Bs = np.empty(instances_shape + (4, 4))
    A2Bs[..., :3, 3] = P[..., :3]
    A2Bs[..., 3, :3] = 0.0
    A2Bs[..., 3, 3] = 1.0

    matrices_from_quaternions(
        P[..., 3:], normalize_quaternions, out=A2Bs[..., :3, :3]
    )

    return A2Bs


def dual_quaternions_from_pqs(pqs):
    """Get dual quaternions from positions and quaternions.

    Parameters
    ----------
    pqs : array-like, shape (..., 7)
        Poses represented by positions and quaternions in the
        order (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    dqs : array, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    pqs = np.asarray(pqs)
    instances_shape = pqs.shape[:-1]
    out = np.empty(instances_shape + (8,))

    # orientation quaternion
    out[..., :4] = pqs[..., 3:]

    # use memory temporarily to store position
    out[..., 4] = 0
    out[..., 5:] = pqs[..., :3]

    out[..., 4:] = 0.5 * batch_concatenate_quaternions(
        out[..., 4:], out[..., :4]
    )
    return out
