"""Representations related to screw theory."""

import numpy as np

from ..batch_rotations import (
    matrices_from_compact_axis_angles,
)
from ..transformations import (
    screw_axis_from_exponential_coordinates,
    screw_parameters_from_screw_axis,
    screw_axis_from_screw_parameters,
)
from ..transformations import (
    transform_from_exponential_coordinates,
)


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
        Stheta_new = (
            screw_axis_from_screw_parameters(q, s_new, h_new) * theta_new
        )
        Sthetas_new[i] = Stheta_new
    return Sthetas_new


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
            axes=screw_axes[..., :3], angles=t, out=A2Bs[..., :3, :3]
        )

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
        A2Bs[..., 0, 3] = (
            -v0 * (o11tms + o22tms - t)
            + v1 * (o01tms + o2cm1)
            + v2 * (o02tms - o1cm1)
        )
        A2Bs[..., 1, 3] = (
            v0 * (o01tms - o2cm1)
            - v1 * (o00tms + o22tms - t)
            + v2 * (o0cm1 + o12tms)
        )
        A2Bs[..., 2, 3] = (
            v0 * (o02tms + o1cm1)
            - v1 * (o0cm1 - o12tms)
            - v2 * (o00tms + o11tms - t)
        )

    A2Bs[ind_only_translation, :3, :3] = np.eye(3)
    A2Bs[ind_only_translation, :3, 3] = Sthetas[ind_only_translation, 3:]

    return A2Bs


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
    h_is_not_inf_mask = ~np.isinf(hs)
    mod_thetas = np.where(h_is_not_inf_mask, thetas, 0.0)
    ds = np.copy(thetas)
    ds[h_is_not_inf_mask] *= hs[h_is_not_inf_mask]

    moments = np.cross(qs, s_axis)
    half_distances = 0.5 * ds
    half_thetas = 0.5 * mod_thetas
    sin_half_angles = np.sin(half_thetas)
    cos_half_angles = np.cos(half_thetas)

    real_w = cos_half_angles
    real_vec = sin_half_angles[..., np.newaxis] * s_axis
    dual_w = -half_distances * sin_half_angles
    dual_vec = (
        sin_half_angles[..., np.newaxis] * moments
        + half_distances[..., np.newaxis]
        * cos_half_angles[..., np.newaxis]
        * s_axis
    )

    result = np.concatenate(
        [real_w[..., np.newaxis], real_vec, dual_w[..., np.newaxis], dual_vec],
        axis=-1,
    )
    return result
