"""Plotting functions."""

import numpy as np

from ._artists import Arrow3D
from ._layout import make_3d_axis
from ..rotations import unitx, unitz, perpendicular_to_vectors, norm_vector


def plot_vector(
    ax=None,
    start=np.zeros(3),
    direction=np.array([1, 0, 0]),
    s=1.0,
    arrowstyle="simple",
    ax_s=1,
    **kwargs,
):
    """Plot Vector.

    Draws an arrow from start to start + s * direction.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    start : array-like, shape (3,), optional (default: [0, 0, 0])
        Start of the vector

    direction : array-like, shape (3,), optional (default: [1, 0, 0])
        Direction of the vector

    s : float, optional (default: 1)
        Scaling of the vector that will be drawn

    arrowstyle : str, or ArrowStyle, optional (default: 'simple')
        See matplotlib's documentation of arrowstyle in
        matplotlib.patches.FancyArrowPatch for more options

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    axis_arrow = Arrow3D(
        [start[0], start[0] + s * direction[0]],
        [start[1], start[1] + s * direction[1]],
        [start[2], start[2] + s * direction[2]],
        mutation_scale=20,
        arrowstyle=arrowstyle,
        **kwargs,
    )
    ax.add_artist(axis_arrow)

    return ax


def plot_length_variable(
    ax=None,
    start=np.zeros(3),
    end=np.ones(3),
    name="l",
    above=False,
    ax_s=1,
    color="k",
    **kwargs,
):
    """Plot length with text at its center.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    start : array-like, shape (3,), optional (default: [0, 0, 0])
        Start point

    end : array-like, shape (3,), optional (default: [1, 1, 1])
        End point

    name : str, optional (default: 'l')
        Text in the middle

    above : bool, optional (default: False)
        Plot name above line

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    color : str, optional (default: black)
        Color in which the cylinder should be plotted

    kwargs : dict, optional (default: {})
        Additional arguments for the text, e.g. fontsize

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    direction = end - start
    length = np.linalg.norm(direction)

    if above:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color,
        )
    else:
        mid1 = start + 0.4 * direction
        mid2 = start + 0.6 * direction
        ax.plot(
            [start[0], mid1[0]],
            [start[1], mid1[1]],
            [start[2], mid1[2]],
            color=color,
        )
        ax.plot(
            [end[0], mid2[0]], [end[1], mid2[1]], [end[2], mid2[2]], color=color
        )

    if np.linalg.norm(direction / length - unitz) < np.finfo(float).eps:
        axis = unitx
    else:
        axis = unitz

    mark = (
        norm_vector(perpendicular_to_vectors(direction, axis)) * 0.03 * length
    )
    mark_start1 = start + mark
    mark_start2 = start - mark
    mark_end1 = end + mark
    mark_end2 = end - mark
    ax.plot(
        [mark_start1[0], mark_start2[0]],
        [mark_start1[1], mark_start2[1]],
        [mark_start1[2], mark_start2[2]],
        color=color,
    )
    ax.plot(
        [mark_end1[0], mark_end2[0]],
        [mark_end1[1], mark_end2[1]],
        [mark_end1[2], mark_end2[2]],
        color=color,
    )
    text_location = start + 0.45 * direction
    if above:
        text_location[2] += 0.3 * length
    ax.text(
        text_location[0],
        text_location[1],
        text_location[2],
        name,
        zdir="x",
        **kwargs,
    )

    return ax
