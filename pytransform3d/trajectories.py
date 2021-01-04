"""Trajectories in three dimensions (position and orientation)."""
import numpy as np
from .plot_utils import Trajectory, make_3d_axis
from .batch_rotations import norm_vectors, matrices_from_quaternions, quaternions_from_matrices, matrix_from_compact_axis_angles


def transforms_from_pqs(P, normalize_quaternions=True):
    """Get sequence of homogeneous matrices from positions and quaternions.

    Parameters
    ----------
    P : array-like, shape (n_steps, 7)
        Sequence of poses represented by positions and quaternions in the
        order (x, y, z, w, vx, vy, vz) for each step

    normalize_quaternions : bool, optional (default: True)
        Normalize quaternions before conversion

    Returns
    -------
    H : array, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices
    """
    P = np.asarray(P)
    H = np.empty((len(P), 4, 4))
    H[:, :3, 3] = P[:, :3]
    H[:, 3, :3] = 0.0
    H[:, 3, 3] = 1.0

    if normalize_quaternions:
        Q = norm_vectors(P[:, 3:])
    else:
        Q = P[:, 3:]

    matrices_from_quaternions(Q, out=H[:, :3, :3])

    return H


matrices_from_pos_quat = transforms_from_pqs


def pqs_from_transforms(H):
    """Get sequence of positions and quaternions from homogeneous matrices.

    Parameters
    ----------
    H : array-like, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices

    Returns
    -------
    P : array, shape (n_steps, 7)
        Sequence of poses represented by positions and quaternions in the
        order (x, y, z, w, vx, vy, vz) for each step
    """
    H = np.asarray(H)
    P = np.empty((len(H), 7))
    P[:, :3] = H[:, :3, 3]
    quaternions_from_matrices(H[:, :3, :3], out=P[:, 3:])
    return P


def transforms_from_exponential_coordinates(Sthetas):
    """TODO"""
    Sthetas = np.asarray(Sthetas)
    instances_shape = Sthetas.shape[:-1]

    thetas = np.linalg.norm(Sthetas[..., :3], axis=-1)

    H = np.empty(instances_shape + (4, 4))
    H[..., 3, :] = (0, 0, 0, 1)
    #H[..., 3, :3] = 0.0
    #H[..., 3, 3] = 1.0

    ind_only_translation = thetas == 0.0
    H[ind_only_translation, :3, :3] = np.eye(3)
    H[ind_only_translation, :3, 3] = Sthetas[ind_only_translation, 3:]
    if np.all(ind_only_translation):
        return H

    ind = thetas != 0.0
    thetas_ind = thetas[ind]
    if instances_shape:
        thetas_ind = thetas_ind.reshape(*instances_shape)
    screw_axes = Sthetas[ind] / thetas_ind
    omega_ind = screw_axes[..., :3]
    v_ind = screw_axes[..., 3:]

    H[ind, :3, :3] = matrix_from_compact_axis_angles(Sthetas[ind, :3])

    # from sympy import *
    # omega0, omega1, omega2, vx, vy, vz, theta = symbols("omega_0 omega_1 omega_2 v_x v_y v_z theta")
    # w = Matrix([[0, -omega2, omega1], [omega2, 0, -omega0], [-omega1, omega0, 0]])
    # v = Matrix([[vx], [vy], [vz]])
    # p = (eye(3) * theta + (1 - cos(theta)) * w + (theta - sin(theta)) * w * w) * v
    # Result:
    # -v_x*(omega_1**2*(theta - sin(theta)) + omega_2**2*(theta - sin(theta)) - theta) + v_y*(omega_0*omega_1*(theta - sin(theta)) + omega_2*(cos(theta) - 1)) + v_z*(omega_0*omega_2*(theta - sin(theta)) - omega_1*(cos(theta) - 1))
    # v_x*(omega_0*omega_1*(theta - sin(theta)) - omega_2*(cos(theta) - 1)) - v_y*(omega_0**2*(theta - sin(theta)) + omega_2**2*(theta - sin(theta)) - theta) + v_z*(omega_0*(cos(theta) - 1) + omega_1*omega_2*(theta - sin(theta)))
    # v_x*(omega_0*omega_2*(theta - sin(theta)) + omega_1*(cos(theta) - 1)) - v_y*(omega_0*(cos(theta) - 1) - omega_1*omega_2*(theta - sin(theta))) - v_z*(omega_0**2*(theta - sin(theta)) + omega_1**2*(theta - sin(theta)) - theta)

    thetas_minus_sin_thetas_ind = thetas_ind - np.sin(thetas_ind)
    cos_thetas_minus_1_ind = np.cos(thetas_ind) - 1.0
    v_ind_0 = v_ind[..., 0]
    v_ind_1 = v_ind[..., 1]
    v_ind_2 = v_ind[..., 2]
    omega_ind_0 = omega_ind[..., 0]
    omega_ind_1 = omega_ind[..., 1]
    omega_ind_2 = omega_ind[..., 2]
    if instances_shape:
        v_ind_0 = v_ind_0.reshape(*instances_shape)
        v_ind_1 = v_ind_1.reshape(*instances_shape)
        v_ind_2 = v_ind_2.reshape(*instances_shape)
        omega_ind_0 = omega_ind_0.reshape(*instances_shape)
        omega_ind_1 = omega_ind_1.reshape(*instances_shape)
        omega_ind_2 = omega_ind_2.reshape(*instances_shape)
    H[ind, 0, 3] = (-v_ind_0 * (omega_ind_1 ** 2 * thetas_minus_sin_thetas_ind
                                 + omega_ind_2 ** 2 * thetas_minus_sin_thetas_ind - thetas_ind)
                    + v_ind_1 * (omega_ind_0 * omega_ind_1 * thetas_minus_sin_thetas_ind
                                  + omega_ind_2 * cos_thetas_minus_1_ind)
                    + v_ind_2 * (omega_ind_0 * omega_ind_2 * thetas_minus_sin_thetas_ind
                                  - omega_ind_1 * cos_thetas_minus_1_ind)).squeeze()
    H[ind, 1, 3] = (v_ind_0 * (omega_ind_0 * omega_ind_1 * thetas_minus_sin_thetas_ind
                                - omega_ind_2 * cos_thetas_minus_1_ind)
                    - v_ind_1 * (omega_ind_0 ** 2 * thetas_minus_sin_thetas_ind
                                  + omega_ind_2 ** 2 * thetas_minus_sin_thetas_ind - thetas_ind)
                    + v_ind_2 * (omega_ind_0 * cos_thetas_minus_1_ind
                                  + omega_ind_1 * omega_ind_2 * thetas_minus_sin_thetas_ind)).squeeze()
    H[ind, 2, 3] = (v_ind_0 * (omega_ind_0 * omega_ind_2 * thetas_minus_sin_thetas_ind
                                + omega_ind_1 * cos_thetas_minus_1_ind)
                    - v_ind_1 * (omega_ind_0 * cos_thetas_minus_1_ind
                                  - omega_ind_1 * omega_ind_2 * thetas_minus_sin_thetas_ind)
                    - v_ind_2 * (omega_ind_0 ** 2 * thetas_minus_sin_thetas_ind
                                  + omega_ind_1 ** 2 * thetas_minus_sin_thetas_ind - thetas_ind)).squeeze()

    return H


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

    H = transforms_from_pqs(P, normalize_quaternions)
    trajectory = Trajectory(H, show_direction, n_frames, s, **kwargs)
    trajectory.add_trajectory(ax)

    return ax
