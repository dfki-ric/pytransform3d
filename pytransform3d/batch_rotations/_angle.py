import numpy as np


def active_matrices_from_angles(basis, angles, out=None):
    """Compute active rotation matrices from rotation about basis vectors.

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angles : array-like, shape (...)
        Rotation angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    angles = np.asarray(angles)
    c = np.cos(angles)
    s = np.sin(angles)

    R_shape = angles.shape + (3, 3)
    if out is None:
        out = np.empty(R_shape)

    out[..., basis, :] = 0.0
    out[..., :, basis] = 0.0
    out[..., basis, basis] = 1.0
    basisp1 = (basis + 1) % 3
    basisp2 = (basis + 2) % 3
    out[..., basisp1, basisp1] = c
    out[..., basisp2, basisp2] = c
    out[..., basisp1, basisp2] = -s
    out[..., basisp2, basisp1] = s

    return out
