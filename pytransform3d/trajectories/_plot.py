from ._pqs import transforms_from_pqs


def plot_trajectory(
    ax=None,
    P=None,
    normalize_quaternions=True,
    show_direction=True,
    n_frames=10,
    s=1.0,
    ax_s=1,
    **kwargs,
):  # pragma: no cover
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
        from ..plot_utils import make_3d_axis

        ax = make_3d_axis(ax_s)

    A2Bs = transforms_from_pqs(P, normalize_quaternions)
    from ..plot_utils import Trajectory

    trajectory = Trajectory(A2Bs, show_direction, n_frames, s, **kwargs)
    trajectory.add_trajectory(ax)

    return ax
