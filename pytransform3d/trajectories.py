"""Trajectories in three dimensions - SE(3).

Conversions from this module operate on batches of poses or transformations
and can be 400 to 1000 times faster than a loop of individual conversions.
"""
import numpy as np
from .batch_rotations import (
    matrices_from_quaternions, quaternions_from_matrices,
    matrices_from_compact_axis_angles, axis_angles_from_matrices,
    batch_concatenate_quaternions, batch_q_conj)
from .transformations import (
    transform_from_exponential_coordinates,
    screw_axis_from_exponential_coordinates, screw_parameters_from_screw_axis,
    screw_axis_from_screw_parameters)
from .rotations import norm_angle


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
        list(range(A2Bs.ndim - 2)) + [A2Bs.ndim - 1, A2Bs.ndim - 2])
    B2As[..., :3, 3] = np.einsum(
        "nij,nj->ni",
        -B2As[..., :3, :3].reshape(-1, 3, 3),
        A2Bs[..., :3, 3].reshape(-1, 3)).reshape(
        *(list(instances_shape) + [3]))
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
    A2Bs : array-like, shape (4, 4)
        Transforms from frame A to frame B

    B2C : array-like, shape (n_transforms, 4, 4)
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
        P[..., 3:], normalize_quaternions, out=A2Bs[..., :3, :3])

    return A2Bs


# DEPRECATED: for backwards compatibility only!
matrices_from_pos_quat = transforms_from_pqs


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
    if instances_shape:
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
    Sthetas[..., 3] = (p0 * ((-o11 - o22) * tan_term + ti)
                       + p1 * (o01 * tan_term + 0.5 * o2)
                       + p2 * (o02 * tan_term - 0.5 * o1)
                       )
    Sthetas[..., 4] = (p0 * (o01 * tan_term - 0.5 * o2)
                       + p1 * ((-o00 - o22) * tan_term + ti)
                       + p2 * (0.5 * o0 + o12 * tan_term)
                       )
    Sthetas[..., 5] = (p0 * (o02 * tan_term + 0.5 * o1)
                       + p1 * (-0.5 * o0 + o12 * tan_term)
                       + p2 * ((-o00 - o11) * tan_term + ti)
                       )

    Sthetas *= thetas[..., np.newaxis]

    ind_only_translation = traces >= 3.0 - np.finfo(float).eps
    Sthetas[ind_only_translation, :3] = 0.0
    Sthetas[ind_only_translation, 3:] = ps[ind_only_translation]

    return Sthetas


def transforms_from_exponential_coordinates(Sthetas):
    """Compute transformations from exponential coordinates.

    Parameters
    ----------
    Sthetas : array-like, shape (..., 6)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    Returns
    -------
    A2Bs : array, shape (..., 4, 4)
        Poses represented by homogeneous matrices
    """
    Sthetas = np.asarray(Sthetas)
    if Sthetas.ndim == 1:
        return transform_from_exponential_coordinates(Sthetas)

    instances_shape = Sthetas.shape[:-1]

    t = np.linalg.norm(Sthetas[..., :3], axis=-1)

    A2Bs = np.empty(instances_shape + (4, 4))
    A2Bs[..., 3, :] = (0, 0, 0, 1)

    ind_only_translation = t == 0.0

    if not np.all(ind_only_translation):
        t[ind_only_translation] = 1.0
        screw_axes = Sthetas / t[..., np.newaxis]

        matrices_from_compact_axis_angles(
            axes=screw_axes[..., :3], angles=t, out=A2Bs[..., :3, :3])

        # from sympy import *
        # o0, o1, o2, vx, vy, vz, t = symbols("o0 o1 o2 v_x v_y v_z t")
        # w = Matrix([[0, -o2, o1], [o2, 0, -o0], [-o1, o0, 0]])
        # v = Matrix([[vx], [vy], [vz]])
        # p = (eye(3) * t + (1 - cos(t)) * w + (t - sin(t)) * w * w) * v
        #
        # Result:
        # -v_x*(o1**2*(t - sin(t)) + o2**2*(t - sin(t)) - t)
        #     + v_y*(o0*o1*(t - sin(t)) + o2*(cos(t) - 1))
        #     + v_z*(o0*o2*(t - sin(t)) - o1*(cos(t) - 1))
        # v_x*(o0*o1*(t - sin(t)) - o2*(cos(t) - 1))
        #     - v_y*(o0**2*(t - sin(t)) + o2**2*(t - sin(t)) - t)
        #     + v_z*(o0*(cos(t) - 1) + o1*o2*(t - sin(t)))
        # v_x*(o0*o2*(t - sin(t)) + o1*(cos(t) - 1))
        #     - v_y*(o0*(cos(t) - 1) - o1*o2*(t - sin(t)))
        #     - v_z*(o0**2*(t - sin(t)) + o1**2*(t - sin(t)) - t)

        tms = t - np.sin(t)
        cm1 = np.cos(t) - 1.0
        o0 = screw_axes[..., 0]
        o1 = screw_axes[..., 1]
        o2 = screw_axes[..., 2]
        v0 = screw_axes[..., 3]
        v1 = screw_axes[..., 4]
        v2 = screw_axes[..., 5]
        o01tms = o0 * o1 * tms
        o12tms = o1 * o2 * tms
        o02tms = o0 * o2 * tms
        o0cm1 = o0 * cm1
        o1cm1 = o1 * cm1
        o2cm1 = o2 * cm1
        o00tms = o0 * o0 * tms
        o11tms = o1 * o1 * tms
        o22tms = o2 * o2 * tms
        v0 = v0.reshape(*instances_shape)
        v1 = v1.reshape(*instances_shape)
        v2 = v2.reshape(*instances_shape)
        A2Bs[..., 0, 3] = (-v0 * (o11tms + o22tms - t)
                           + v1 * (o01tms + o2cm1)
                           + v2 * (o02tms - o1cm1))
        A2Bs[..., 1, 3] = (v0 * (o01tms - o2cm1)
                           - v1 * (o00tms + o22tms - t)
                           + v2 * (o0cm1 + o12tms))
        A2Bs[..., 2, 3] = (v0 * (o02tms + o1cm1)
                           - v1 * (o0cm1 - o12tms)
                           - v2 * (o00tms + o11tms - t))

    A2Bs[ind_only_translation, :3, :3] = np.eye(3)
    A2Bs[ind_only_translation, :3, 3] = Sthetas[ind_only_translation, 3:]

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
    out = np.empty(list(instances_shape) + [8])

    # orientation quaternion
    out[..., :4] = pqs[..., 3:]

    # use memory temporarily to store position
    out[..., 4] = 0
    out[..., 5:] = pqs[..., :3]

    out[..., 4:] = 0.5 * batch_concatenate_quaternions(
        out[..., 4:], out[..., :4])
    return out


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
    out = np.empty(list(instances_shape) + [8])

    # orientation quaternion
    out[..., :4] = quaternions_from_matrices(A2Bs[..., :3, :3])

    # use memory temporarily to store position
    out[..., 4] = 0
    out[..., 5:] = A2Bs[..., :3, 3]

    out[..., 4:] = 0.5 * batch_concatenate_quaternions(
        out[..., 4:], out[..., :4])
    return out


def pqs_from_dual_quaternions(dqs):
    """Get positions and quaternions from dual quaternions.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    pqs : array, shape (..., 7)
        Poses represented by positions and quaternions in the
        order (x, y, z, qw, qx, qy, qz)
    """
    dqs = np.asarray(dqs)
    instances_shape = dqs.shape[:-1]
    out = np.empty(list(instances_shape) + [7])
    out[..., 3:] = dqs[..., :4]
    out[..., :3] = 2 * batch_concatenate_quaternions(
        dqs[..., 4:], batch_q_conj(out[..., 3:]))[..., 1:]
    return out



def norm_axis_angles(a):
    """Normalize axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (..., 4)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array, shape (..., 4)
        Axis of rotation and rotation angle: (x, y, z, angle). The length
        of the axis vector is 1 and the angle is in [0, pi). No rotation
        is represented by [1, 0, 0, 0].
    """
    angles = a[..., 3]
    norm = np.linalg.norm(a[..., :3], axis=-1)


    res = np.ones_like(a)

    # Create masks for elements where angle or norm is zero
    zero_mask = (angles == 0.0) | (norm == 0.0)


    non_zero_mask = ~zero_mask
    res[non_zero_mask, :3] = (
        a[non_zero_mask, :3] / norm[non_zero_mask, np.newaxis]
    )


    angle_normalized = norm_angle(angles)

    negative_angle_mask = angle_normalized < 0.0
    res[negative_angle_mask, :3] *= -1.0
    angle_normalized[negative_angle_mask] *= -1.0

    res[non_zero_mask, 3] = angle_normalized[non_zero_mask]
    return res


def axis_angles_from_quaternions(qs):
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    We usually assume active rotations.

    Parameters
    ----------
    q : array-like, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array, shape (..., 4)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    quaternion_vector_part = qs[..., 1:]
    p_norm = np.linalg.norm(quaternion_vector_part, axis=-1)

    # Create a mask for quaternions where p_norm is small
    small_p_norm_mask = p_norm < np.finfo(float).eps

    # Initialize the output with default values for small p_norm cases
    result = np.zeros_like(qs)
    result[small_p_norm_mask] = np.array([1.0, 0.0, 0.0, 0.0])

    # For non-zero norms, calculate axis, clamped w, and angle
    non_zero_mask = ~small_p_norm_mask
    axis = np.zeros_like(quaternion_vector_part)
    axis[non_zero_mask] = (
        quaternion_vector_part[non_zero_mask] /
        p_norm[non_zero_mask, np.newaxis]
    )


    w_clamped = np.clip(qs[..., 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w_clamped)

    # Stack the axis and the angle together a
    # and normalize the axis-angle representation
    result[non_zero_mask] = norm_axis_angles(
        np.hstack((
            axis[non_zero_mask],
            angle[non_zero_mask, np.newaxis]
        ))
    )

    return result


def screw_parameters_from_dual_quaternions(dqs):
    """Compute screw parameters from dual quaternions.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    qs : array, shape (..., 3)
        Vector to a point on the screw axis

    s_axiss : array, shape (..., 3)
        Direction vector of the screw axis

    hs : array, shape (...,)
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    thetas : array, shape (...,)
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    See Also
    --------
    pytransform3d.transformations.screw_parameters_from_dual_quaternion :
        Compute screw parameters from dual quaternion.
    """
    reals = dqs[..., :4]
    duals = dqs[..., 4:]

    a = axis_angles_from_quaternions(reals)
    s_axis = a[..., :3]
    thetas = a[..., 3]

    translation = 2 * batch_concatenate_quaternions(
        duals, batch_q_conj(reals))[..., 1:]

    # instead of the if/else stamenets in the 
    # screw_parameters_from_dual_quaternion function
    # we use mask array to enable vectorized operations
    # the name of the mask represent the according block in
    # the original function
    outer_if_mask = np.abs(thetas) < np.finfo(float).eps
    outer_else_mask = np.logical_not(outer_if_mask)


    ds = np.linalg.norm(translation, axis=1)
    inner_if_mask = ds < np.finfo(float).eps

    outer_if_inner_if_mask = np.logical_and(outer_if_mask, inner_if_mask)
    outer_if_inner_else_mask = np.logical_and(outer_if_mask,
                                              np.logical_not(inner_if_mask))

    # Initialize the outputs
    qs = np.zeros((dqs.shape[0], 3))
    thetas[outer_if_mask] = ds[outer_if_mask]
    hs = np.full(dqs.shape[0], np.inf)

    if np.any(outer_if_inner_if_mask):
        s_axis[outer_if_inner_if_mask] = np.array([1, 0, 0])

    if np.any(outer_if_inner_else_mask):
        s_axis[outer_if_inner_else_mask] = (
            translation[outer_if_inner_else_mask]
            / ds[outer_if_inner_else_mask][..., np.newaxis]
        )

    qs = np.zeros((dqs.shape[0], 3))
    thetas[outer_if_mask] = ds[outer_if_mask]
    hs = np.full(dqs.shape[0], np.inf)

    if np.any(outer_else_mask):
        distance = np.einsum('ij,ij->i',
                             translation[outer_else_mask],
                             s_axis[outer_else_mask])

        moment = 0.5 * (
            np.cross(translation[outer_else_mask], s_axis[outer_else_mask]) +
            (translation[outer_else_mask] - distance[..., np.newaxis] *
             s_axis[outer_else_mask])
            / np.tan(0.5 * thetas[outer_else_mask])[..., np.newaxis]
        )

        qs[outer_else_mask] = np.cross(s_axis[outer_else_mask], moment)
        hs[outer_else_mask] = distance / thetas[outer_else_mask]

    return qs, s_axis, hs, thetas


def dual_quaternions_from_screw_parameters(qs, s_axis, hs, thetas):
    """Compute dual quaternions from arrays of screw parameters.

    Parameters
    ----------
    qs : array-like, shape (..., 3)
        Vector to a point on the screw axis

    s_axis : array-like, shape (..., 3)
        Direction vector of the screw axis

    hs : array-like, shape (...,)
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    thetas : array-like, shape (...,)
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    Returns
    -------
    dqs : array, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    See Also
    --------
    pytransform3d.transformations.dual_quaternion_from_screw_parameters :
        Compute dual quaternion from screw parameters.
    """
    ds = np.zeros_like(hs)

    h_is_inf_mask = np.isinf(hs)
    h_is_not_inf_mask = np.logical_not(h_is_inf_mask)

    mod_thetas = thetas.copy()
    if np.any(h_is_inf_mask):
        ds[h_is_inf_mask] = thetas[h_is_inf_mask]
        mod_thetas[h_is_inf_mask] = 0

    if np.any(h_is_not_inf_mask):
        ds[h_is_not_inf_mask] = hs[h_is_not_inf_mask] * \
            thetas[h_is_not_inf_mask]

    moments = np.cross(qs, s_axis)
    half_distances = 0.5 * ds
    half_thetas = 0.5 * mod_thetas
    sin_half_angles = np.sin(0.5 * half_thetas)
    cos_half_angles = np.cos(0.5 * half_thetas)

    real_w = cos_half_angles
    real_vec = sin_half_angles[..., np.newaxis] * s_axis
    dual_w = -half_distances * sin_half_angles
    dual_vec = (sin_half_angles[..., np.newaxis] * moments +
                half_distances[..., np.newaxis] *
                cos_half_angles[..., np.newaxis] * s_axis)

    result = np.concatenate([real_w[..., np.newaxis],
                             real_vec, dual_w[..., np.newaxis],
                             dual_vec], axis=-1)
    return result


def dual_quaternions_power(dqs, ts):
    """Compute power of unit dual quaternions with respect to scalar.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternions to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    t : array-like, shape (...)
        Exponent

    Returns
    -------
    dq_ts : array, shape (..., 8)
        Unit dual quaternions to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz) ** t

    See Also
    --------
    pytransform3d.transformations.dual_quaternion_power :
        Compute power of unit dual quaternion with respect to scalar.
    """
    q, s_axis, h, theta = screw_parameters_from_dual_quaternions(dqs)
    return dual_quaternions_from_screw_parameters(q, s_axis, h, theta * ts)


def dual_quaternions_sclerp(starts, ends, ts):
    """Screw linear interpolation (ScLERP) for arry of dual quaternions.

    Parameters
    ----------
    starts : array-like, shape (..., 8)
        Unit dual quaternion to represent start poses:
        (pw, px, py, pz, qw, qx, qy, qz)

    end : array-like, shape (..., 8)
        Unit dual quaternion to represent end poses:
        (pw, px, py, pz, qw, qx, qy, qz)

    ts : array-like, shape (...)
        Positions between starts and goals

    Returns
    -------
    dq_ts : array, shape (..., 8)
        Interpolated unit dual quaternion: (pw, px, py, pz, qw, qx, qy, qz)


    See Also
    --------
    pytransform3d.transformations.dual_quaternion_sclerp :
        Screw linear interpolation (ScLERP) for dual quaternions.
    """
    if starts.shape != ends.shape:
        raise ValueError(
            "The 'starts' and 'ends' arrays must have the same shape."
        )

    if starts.shape[0] != ts.shape[0]:
        raise ValueError(
            "ts array must have the same number of elements as starts array"
        )

    diffs = batch_concatenate_dual_quaternions(batch_dq_q_conj(starts), ends)
    powers = dual_quaternions_power(diffs, ts)
    return batch_concatenate_dual_quaternions(starts, powers)


def transforms_from_dual_quaternions(dqs):
    """Get transformations from dual quaternions.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    A2Bs : array, shape (..., 4, 4)
        Poses represented by homogeneous matrices
    """
    dqs = np.asarray(dqs)
    instances_shape = dqs.shape[:-1]
    out = np.empty(list(instances_shape) + [4, 4])
    out[..., :3, :3] = matrices_from_quaternions(dqs[..., :4])
    out[..., :3, 3] = 2 * batch_concatenate_quaternions(
        dqs[..., 4:], batch_q_conj(dqs[..., :4]))[..., 1:]
    out[..., 3, :3] = 0.0
    out[..., 3, 3] = 1.0
    return out


def batch_dq_conj(dqs):
    """Conjugate of dual quaternions.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, -qw, qx, qy, qz). It is a combination of the quaternion
    conjugate and the dual number conjugate.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_conjugates : array-like, shape (..., 8)
        Conjugates of dual quaternions: (pw, -px, -py, -pz, -qw, qx, qy, qz)
    """
    out = np.empty_like(dqs)
    out[..., 0] = dqs[..., 0]
    out[..., 1:5] = -dqs[..., 1:5]
    out[..., 5:] = dqs[..., 5:]
    return out


def batch_dq_q_conj(dqs):
    """Quaternion conjugate of dual quaternions.

    For unit dual quaternions that represent transformations,
    this function is equivalent to the inverse of the
    corresponding transformation matrix.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_q_conjugates : array, shape (..., 8)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, qw, -qx, -qy, -qz)

    See Also
    --------
    pytransform3d.transformations.dq_q_conj
        Quaternion conjugate of dual quaternions.
    """
    out = np.empty_like(dqs)
    out[..., 0] = dqs[..., 0]
    out[..., 1:4] = -dqs[..., 1:4]
    out[..., 4] = dqs[..., 4]
    out[..., 5:8] = -dqs[..., 5:8]
    return out


def batch_concatenate_dual_quaternions(dqs1, dqs2):
    """Concatenate dual quaternions.

    Suppose we want to apply two extrinsic transforms given by dual
    quaternions dq1 and dq2 to a vector v. We can either apply dq2 to v and
    then dq1 to the result or we can concatenate dq1 and dq2 and apply the
    result to v.

    Parameters
    ----------
    dqs1 : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    dqs2 : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dqs3 : array, shape (8,)
        Products of the two batches of dual quaternions:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    dqs1 = np.asarray(dqs1)
    dqs2 = np.asarray(dqs2)

    out = np.empty_like(dqs1)
    out[..., :4] = batch_concatenate_quaternions(
        dqs1[..., :4], dqs2[..., :4])
    out[..., 4:] = (
        batch_concatenate_quaternions(dqs1[..., :4], dqs2[..., 4:]) +
        batch_concatenate_quaternions(dqs1[..., 4:], dqs2[..., :4]))
    return out


def batch_dq_prod_vector(dqs, V):
    """Apply transforms represented by a dual quaternions to vectors.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternions

    V : array-like, shape (..., 3)
        3d vectors

    Returns
    -------
    W : array, shape (3,)
        3d vectors
    """
    dqs = np.asarray(dqs)

    v_dqs = np.empty_like(dqs)
    v_dqs[..., 0] = 1.0
    v_dqs[..., 1:5] = 0.0
    v_dqs[..., 5:] = V
    v_dq_transformed = batch_concatenate_dual_quaternions(
        batch_concatenate_dual_quaternions(dqs, v_dqs),
        batch_dq_conj(dqs))
    return v_dq_transformed[5:]


def plot_trajectory(
        ax=None, P=None, normalize_quaternions=True, show_direction=True,
        n_frames=10, s=1.0, ax_s=1, **kwargs):  # pragma: no cover
    """Plot pose trajectory.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    P : array-like, shape (n_steps, 7), optional (default: None)
        Sequence of poses represented by positions and quaternions in the
        order (x, y, z, w, vx, vy, vz) for each step

    normalize_quaternions : bool, optional (default: True)
        Normalize quaternions before plotting

    show_direction : bool, optional (default: True)
        Plot an arrow to indicate the direction of the trajectory

    n_frames : int, optional (default: 10)
        Number of frames that should be plotted to indicate the rotation

    s : float, optional (default: 1)
        Scaling of the frames that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis

    Raises
    ------
    ValueError
        If trajectory does not contain any elements.
    """
    if P is None or len(P) == 0:
        raise ValueError("Trajectory does not contain any elements.")

    if ax is None:
        from .plot_utils import make_3d_axis
        ax = make_3d_axis(ax_s)

    A2Bs = transforms_from_pqs(P, normalize_quaternions)
    from .plot_utils import Trajectory
    trajectory = Trajectory(A2Bs, show_direction, n_frames, s, **kwargs)
    trajectory.add_trajectory(ax)

    return ax


def mirror_screw_axis_direction(Sthetas):
    """Switch to the other representation of the same transformation.

    We take the negative of the screw axis, invert the rotation angle
    and adapt the screw pitch accordingly. For this operation we have
    to convert exponential coordinates to screw parameters first.

    Parameters
    ----------
    Sthetas : array-like, shape (n_steps, 6)
        Exponential coordinates of transformation:
        (omega_x, omega_y, omega_z, v_x, v_y, v_z)

    Returns
    -------
    Sthetas : array, shape (n_steps, 6)
        Exponential coordinates of transformation:
        (omega_x, omega_y, omega_z, v_x, v_y, v_z)
    """
    Sthetas_new = np.empty((len(Sthetas), 6))
    for i, Stheta in enumerate(Sthetas):
        S, theta = screw_axis_from_exponential_coordinates(Stheta)
        q, s, h = screw_parameters_from_screw_axis(S)
        s_new = -s
        theta_new = 2.0 * np.pi - theta
        h_new = -h * theta / theta_new
        Stheta_new = screw_axis_from_screw_parameters(
            q, s_new, h_new) * theta_new
        Sthetas_new[i] = Stheta_new
    return Sthetas_new
