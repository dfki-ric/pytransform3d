"""Trajectories in three dimensions (position and orientation)."""
import numpy as np
from .plot_utils import Trajectory, make_3d_axis
from .batch_rotations import norm_vectors, matrices_from_quaternions, quaternions_from_matrices


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
