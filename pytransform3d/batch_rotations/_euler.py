import numpy as np

from ._angle import active_matrices_from_angles


def active_matrices_from_intrinsic_euler_angles(
    basis1, basis2, basis3, e, out=None
):
    """Compute active rotation matrices from intrinsic Euler angles.

    Parameters
    ----------
    basis1 : int
        Basis vector of first rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis2 : int
        Basis vector of second rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis3 : int
        Basis vector of third rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    e : array-like, shape (..., 3)
        Euler angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    e = np.asarray(e)
    R_shape = e.shape + (3,)
    R_alpha = active_matrices_from_angles(basis1, e[..., 0].flat)
    R_beta = active_matrices_from_angles(basis2, e[..., 1].flat)
    R_gamma = active_matrices_from_angles(basis3, e[..., 2].flat)

    if out is None:
        out = np.empty(R_shape)

    out[:] = np.einsum(
        "nij,njk->nik", np.einsum("nij,njk->nik", R_alpha, R_beta), R_gamma
    ).reshape(R_shape)

    return out


def active_matrices_from_extrinsic_euler_angles(
    basis1, basis2, basis3, e, out=None
):
    """Compute active rotation matrices from extrinsic Euler angles.

    Parameters
    ----------
    basis1 : int
        Basis vector of first rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis2 : int
        Basis vector of second rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis3 : int
        Basis vector of third rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    e : array-like, shape (..., 3)
        Euler angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    e = np.asarray(e)
    R_shape = e.shape + (3,)
    R_alpha = active_matrices_from_angles(basis1, e[..., 0].flat)
    R_beta = active_matrices_from_angles(basis2, e[..., 1].flat)
    R_gamma = active_matrices_from_angles(basis3, e[..., 2].flat)

    if out is None:
        out = np.empty(R_shape)

    out[:] = np.einsum(
        "nij,njk->nik", np.einsum("nij,njk->nik", R_gamma, R_beta), R_alpha
    ).reshape(R_shape)

    return out
