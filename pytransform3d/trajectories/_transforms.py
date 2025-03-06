import numpy as np

from pytransform3d.batch_rotations import (
    quaternions_from_matrices,
    axis_angles_from_matrices,
    batch_concatenate_quaternions,
)


def invert_transforms(A2Bs):
    """Invert transforms.

    Parameters
    ----------
    A2Bs : array-like, shape (..., 4, 4)
        Transforms from frames A to frames B

    Returns
    -------
    B2As : array, shape (..., 4, 4)
        Transforms from frames B to frames A

    See Also
    --------
    pytransform3d.transformations.invert_transform :
        Invert one transformation.
    """
    A2Bs = np.asarray(A2Bs)
    instances_shape = A2Bs.shape[:-2]
    B2As = np.empty_like(A2Bs)
    # ( R t )^-1   ( R^T -R^T*t )
    # ( 0 1 )    = ( 0    1     )
    B2As[..., :3, :3] = A2Bs[..., :3, :3].transpose(
        tuple(range(A2Bs.ndim - 2)) + (A2Bs.ndim - 1, A2Bs.ndim - 2)
    )
    B2As[..., :3, 3] = np.einsum(
        "nij,nj->ni",
        -B2As[..., :3, :3].reshape(-1, 3, 3),
        A2Bs[..., :3, 3].reshape(-1, 3),
    ).reshape(*(instances_shape + (3,)))
    B2As[..., 3, :3] = 0.0
    B2As[..., 3, 3] = 1.0
    return B2As


def concat_one_to_many(A2B, B2Cs):
    """Concatenate transformation A2B with multiple transformations B2C.

    We use the extrinsic convention, which means that B2Cs are left-multiplied
    to A2B.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    B2Cs : array-like, shape (n_transforms, 4, 4)
        Transforms from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_many_to_one :
        Concatenate multiple transformations with one.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    return np.einsum("nij,jk->nik", B2Cs, A2B)


def concat_many_to_one(A2Bs, B2C):
    """Concatenate multiple transformations A2B with transformation B2C.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2Bs.

    Parameters
    ----------
    A2Bs : array-like, shape (n_transforms, 4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (4, 4)
        Transform from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_one_to_many :
        Concatenate one transformation with multiple transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    return np.einsum("ij,njk->nik", B2C, A2Bs)


def concat_many_to_many(A2B, B2C):
    """Concatenate multiple transformations A2B with transformation B2C.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2Bs.

    Parameters
    ----------
    A2B : array-like, shape (n_transforms, 4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (n_transforms, 4, 4)
        Transform from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_many_to_one :
        Concatenate one transformation with multiple transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    return np.einsum("ijk,ikl->ijl", B2C, A2B)


def concat_dynamic(A2B, B2C):
    """Concatenate multiple transformations A2B,B2C with different shapes.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2Bs. it can handle different shapes of A2B and B2C dynamically.

    Parameters
    ----------
    A2B : array-like, shape (n_transforms, 4, 4) or (4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (n_transforms, 4, 4) or (4, 4)
        Transform from frame B to frame C

    Returns
    -------
    A2Cs : array, shape (n_transforms, 4, 4) or (4, 4)
        Transforms from frame A to frame C

    See Also
    --------
    concat_many_to_one :
        Concatenate multiple transformations with one transformation.

    concat_one_to_many :
        Concatenate one transformation with multiple transformations.

    concat_many_to_many :
        Concatenate multiple transformations with multiple transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.
    """
    A2B = np.asarray(A2B)
    B2C = np.asarray(B2C)
    if B2C.ndim == 2 and A2B.ndim == 2:
        return B2C.dot(A2B)
    elif B2C.ndim == 2 and A2B.ndim == 3:
        return concat_many_to_one(A2B, B2C)
    elif B2C.ndim == 3 and A2B.ndim == 2:
        return concat_one_to_many(A2B, B2C)
    elif B2C.ndim == 3 and A2B.ndim == 3:
        return concat_many_to_many(A2B, B2C)
    else:
        raise ValueError(
            "Expected ndim 2 or 3; got B2C.ndim=%d, A2B.ndim=%d"
            % (B2C.ndim, A2B.ndim)
        )


def pqs_from_transforms(A2Bs):
    """Get sequence of positions and quaternions from homogeneous matrices.

    Parameters
    ----------
    A2Bs : array-like, shape (..., 4, 4)
        Poses represented by homogeneous matrices

    Returns
    -------
    P : array, shape (n_steps, 7)
        Poses represented by positions and quaternions in the
        order (x, y, z, qw, qx, qy, qz) for each step
    """
    A2Bs = np.asarray(A2Bs)
    instances_shape = A2Bs.shape[:-2]
    P = np.empty(instances_shape + (7,))
    P[..., :3] = A2Bs[..., :3, 3]
    quaternions_from_matrices(A2Bs[..., :3, :3], out=P[..., 3:])
    return P


def exponential_coordinates_from_transforms(A2Bs):
    """Compute exponential coordinates from transformations.

    Parameters
    ----------
    A2Bs : array-like, shape (..., 4, 4)
        Poses represented by homogeneous matrices

    Returns
    -------
    Sthetas : array, shape (..., 6)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.
    """
    A2Bs = np.asarray(A2Bs)

    instances_shape = A2Bs.shape[:-2]

    Rs = A2Bs[..., :3, :3]
    ps = A2Bs[..., :3, 3]

    traces = np.einsum("nii", Rs.reshape(-1, 3, 3))
    if instances_shape:  # noqa: SIM108
        traces = traces.reshape(*instances_shape)
    else:
        # this works because indX will be a single boolean and
        # out[True, n] = value will assign value to out[n], while
        # out[False, n] = value will not assign value to out[n]
        traces = traces[0]

    Sthetas = np.empty(instances_shape + (6,))

    omega_thetas = axis_angles_from_matrices(Rs, traces=traces)
    Sthetas[..., :3] = omega_thetas[..., :3]
    thetas = omega_thetas[..., 3]

    # from sympy import *
    # o0, o1, o2, px, py, pz, t = symbols("o0 o1 o2 p0 p1 p2 theta")
    # w = Matrix([[0, -o2, o1], [o2, 0, -o0], [-o1, o0, 0]])
    # p = Matrix([[px], [py], [pz]])
    # v = (eye(3) / t - 0.5 * w + (1 / t - 0.5 / tan(t / 2.0)) * w * w) * p

    # Result:
    # p0*(-o1**2*(-0.5/tan(0.5*t) + 1/t)
    #     - o2**2*(-0.5/tan(0.5*t) + 1/t) + 1/t)
    #     + p1*(o0*o1*(-0.5/tan(0.5*t) + 1/t) + 0.5*o2)
    #     + p2*(o0*o2*(-0.5/tan(0.5*t) + 1/t) - 0.5*o1)
    # p0*(o0*o1*(-0.5/tan(0.5*t) + 1/t) - 0.5*o2)
    #     + p1*(-o0**2*(-0.5/tan(0.5*t) + 1/t)
    #           - o2**2*(-0.5/tan(0.5*t) + 1/t) + 1/t)
    #     + p2*(0.5*o0 + o1*o2*(-0.5/tan(0.5*t) + 1/t))
    # p0*(o0*o2*(-0.5/tan(0.5*t) + 1/t) + 0.5*o1)
    #     + p1*(-0.5*o0 + o1*o2*(-0.5/tan(0.5*t) + 1/t))
    #     + p2*(-o0**2*(-0.5/tan(0.5*t) + 1/t)
    #           - o1**2*(-0.5/tan(0.5*t) + 1/t) + 1/t)

    thetas = np.maximum(thetas, np.finfo(float).tiny)
    ti = 1.0 / thetas
    tan_term = -0.5 / np.tan(thetas / 2.0) + ti
    o0 = omega_thetas[..., 0]
    o1 = omega_thetas[..., 1]
    o2 = omega_thetas[..., 2]
    p0 = ps[..., 0]
    p1 = ps[..., 1]
    p2 = ps[..., 2]
    o00 = o0 * o0
    o01 = o0 * o1
    o02 = o0 * o2
    o11 = o1 * o1
    o12 = o1 * o2
    o22 = o2 * o2
    Sthetas[..., 3] = (
        p0 * ((-o11 - o22) * tan_term + ti)
        + p1 * (o01 * tan_term + 0.5 * o2)
        + p2 * (o02 * tan_term - 0.5 * o1)
    )
    Sthetas[..., 4] = (
        p0 * (o01 * tan_term - 0.5 * o2)
        + p1 * ((-o00 - o22) * tan_term + ti)
        + p2 * (0.5 * o0 + o12 * tan_term)
    )
    Sthetas[..., 5] = (
        p0 * (o02 * tan_term + 0.5 * o1)
        + p1 * (-0.5 * o0 + o12 * tan_term)
        + p2 * ((-o00 - o11) * tan_term + ti)
    )

    Sthetas *= thetas[..., np.newaxis]

    ind_only_translation = traces >= 3.0 - np.finfo(float).eps
    Sthetas[ind_only_translation, :3] = 0.0
    Sthetas[ind_only_translation, 3:] = ps[ind_only_translation]

    return Sthetas


def dual_quaternions_from_transforms(A2Bs):
    """Get dual quaternions from transformations.

    Parameters
    ----------
    A2Bs : array-like, shape (..., 4, 4)
        Poses represented by homogeneous matrices

    Returns
    -------
    dqs : array, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    A2Bs = np.asarray(A2Bs)
    instances_shape = A2Bs.shape[:-2]
    out = np.empty(instances_shape + (8,))

    # orientation quaternion
    out[..., :4] = quaternions_from_matrices(A2Bs[..., :3, :3])

    # use memory temporarily to store position
    out[..., 4] = 0
    out[..., 5:] = A2Bs[..., :3, 3]

    out[..., 4:] = 0.5 * batch_concatenate_quaternions(
        out[..., 4:], out[..., :4]
    )
    return out
