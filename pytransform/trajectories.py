import numpy as np
from .plot_utils import Arrow3D, make_3d_axis
from .rotations import matrix_from_quaternion, plot_basis


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


def plot_trajectory(ax=None, P=None, show_direction=True, n_frames=10,
                    s=1.0, ax_s=1, **kwargs):
    """Plot pose trajectory.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    P : array-like, shape (n_steps, 7), optional (default: None)
        Sequence of poses represented by positions and quaternions in the
        order (x, y, z, w, vx, vy, vz) for each step

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

    ax.plot(P[:, 0], P[:, 1], P[:, 2], **kwargs)

    key_frames = np.linspace(0, P.shape[0] - 1, n_frames).astype(np.int)
    for p in P[key_frames]:
        plot_basis(ax, matrix_from_quaternion(p[3:]), p[:3], s=s, **kwargs)

    if show_direction:
        start = 0.8 * P[0, :3] + 0.2 * P[-1, :3]
        end = 0.2 * P[0, :3] + 0.8 * P[-1, :3]
        direction_arrow = Arrow3D(
            [start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            mutation_scale=20, lw=1, arrowstyle="-|>", color="k")
        ax.add_artist(direction_arrow)

    return ax
