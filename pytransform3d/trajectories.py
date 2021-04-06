"""Trajectories in three dimensions - SE(3).

Conversions from this module operate on batches of poses or transformations
and can be 400 to 1000 times faster than a loop of individual conversions.
"""
import numpy as np
from .plot_utils import Trajectory, make_3d_axis
from .batch_rotations import (
    matrices_from_quaternions, quaternions_from_matrices,
    matrices_from_compact_axis_angles, axis_angles_from_matrices,
    batch_concatenate_quaternions, batch_q_conj)
from .transformations import transform_from_exponential_coordinates


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
    # omega0, omega1, omega2, px, py, pz, theta = symbols("o0 o1 o2 p0 p1 p2 theta")
    # w = Matrix([[0, -omega2, omega1], [omega2, 0, -omega0], [-omega1, omega0, 0]])
    # p = Matrix([[px], [py], [pz]])
    # v = (eye(3) / theta - 0.5 * w + (1.0 / theta - 0.5 / tan(theta / 2.0)) * w * w) * p

    # Result:
    # p0*(-o1**2*(-0.5/tan(0.5*theta) + 1.0/theta) - o2**2*(-0.5/tan(0.5*theta) + 1.0/theta) + 1/theta)
    #     + p1*(o0*o1*(-0.5/tan(0.5*theta) + 1.0/theta) + 0.5*o2)
    #     + p2*(o0*o2*(-0.5/tan(0.5*theta) + 1.0/theta) - 0.5*o1)
    # p0*(o0*o1*(-0.5/tan(0.5*theta) + 1.0/theta) - 0.5*o2)
    #     + p1*(-o0**2*(-0.5/tan(0.5*theta) + 1.0/theta) - o2**2*(-0.5/tan(0.5*theta) + 1.0/theta) + 1/theta)
    #     + p2*(0.5*o0 + o1*o2*(-0.5/tan(0.5*theta) + 1.0/theta))
    # p0*(o0*o2*(-0.5/tan(0.5*theta) + 1.0/theta) + 0.5*o1)
    #     + p1*(-0.5*o0 + o1*o2*(-0.5/tan(0.5*theta) + 1.0/theta))
    #     + p2*(-o0**2*(-0.5/tan(0.5*theta) + 1.0/theta) - o1**2*(-0.5/tan(0.5*theta) + 1.0/theta) + 1/theta)

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
        # omega0, omega1, omega2, vx, vy, vz, theta = symbols("omega_0 omega_1 omega_2 v_x v_y v_z theta")
        # w = Matrix([[0, -omega2, omega1], [omega2, 0, -omega0], [-omega1, omega0, 0]])
        # v = Matrix([[vx], [vy], [vz]])
        # p = (eye(3) * theta + (1 - cos(theta)) * w + (theta - sin(theta)) * w * w) * v
        #
        # Result:
        # -v_x*(omega_1**2*(theta - sin(theta)) + omega_2**2*(theta - sin(theta)) - theta)
        #     + v_y*(omega_0*omega_1*(theta - sin(theta)) + omega_2*(cos(theta) - 1))
        #     + v_z*(omega_0*omega_2*(theta - sin(theta)) - omega_1*(cos(theta) - 1))
        # v_x*(omega_0*omega_1*(theta - sin(theta)) - omega_2*(cos(theta) - 1))
        #     - v_y*(omega_0**2*(theta - sin(theta)) + omega_2**2*(theta - sin(theta)) - theta)
        #     + v_z*(omega_0*(cos(theta) - 1) + omega_1*omega_2*(theta - sin(theta)))
        # v_x*(omega_0*omega_2*(theta - sin(theta)) + omega_1*(cos(theta) - 1))
        #     - v_y*(omega_0*(cos(theta) - 1) - omega_1*omega_2*(theta - sin(theta)))
        #     - v_z*(omega_0**2*(theta - sin(theta)) + omega_1**2*(theta - sin(theta)) - theta)

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


def batch_dq_conj(dqs):
    """TODO"""
    out = np.empty_like(dqs)
    out[..., 0] = dqs[..., 0]
    out[..., 1:5] = -dqs[..., 1:5]
    out[..., 5:] = dqs[..., 5:]
    return out


def dual_quaternions_from_pqs(pqs):
    """TODO"""
    instances_shape = pqs.shape[:-1]
    out = np.empty(list(instances_shape) + [8])
    out[..., :4] = pqs[..., 3:]
    # use memory temporarily to store position
    out[..., 4] = 0
    out[..., 5:] = pqs[..., :3]
    out[..., 4:] = 0.5 * batch_concatenate_quaternions(
        out[..., 4:], out[..., :4])
    return out


def pqs_from_dual_quaternions(dqs):
    """TODO"""
    instances_shape = dqs.shape[:-1]
    out = np.empty(list(instances_shape) + [7])
    out[..., 3:] = dqs[..., :4]
    out[..., :3] = 2 * batch_concatenate_quaternions(
        dqs[..., 4:], batch_q_conj(out[..., 3:]))[..., 1:]
    return out


def batch_concatenate_dual_quaternions(dqs1, dqs2):
    """TODO"""
    out = np.empty_like(dqs1)
    out[..., :4] = batch_concatenate_quaternions(dqs1[:4], dqs2[:4])
    out[..., 4:] = (batch_concatenate_quaternions(dqs1[:4], dqs2[4:]) +
                    batch_concatenate_quaternions(dqs1[4:], dqs2[:4]))
    return out


def batch_dq_prod_vector(dqs, v):
    """TODO"""
    v_dqs = np.empty_like(dqs)
    v_dqs[..., 0] = 1.0
    v_dqs[..., 1:5] = 0.0
    v_dqs[..., 5:] = v
    v_dq_transformed = batch_concatenate_dual_quaternions(
        batch_concatenate_quaternions(dqs, v_dqs),
        batch_dq_conj(dqs))
    return v_dq_transformed[5:]


def plot_trajectory(ax=None, P=None, normalize_quaternions=True, show_direction=True, n_frames=10, s=1.0, ax_s=1, **kwargs):
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
    """
    if P is None or len(P) == 0:
        raise ValueError("Trajectory does not contain any elements.")

    if ax is None:
        ax = make_3d_axis(ax_s)

    A2Bs = transforms_from_pqs(P, normalize_quaternions)
    trajectory = Trajectory(A2Bs, show_direction, n_frames, s, **kwargs)
    trajectory.add_trajectory(ax)

    return ax
