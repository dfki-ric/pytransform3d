"""Plotting functions."""
import numpy as np
from ._utils import (check_matrix, check_axis_angle,
                     perpendicular_to_vectors, angle_between_vectors)
from ._constants import a_id, p0, unitx, unity
from ._slerp import slerp_weights


def plot_basis(ax=None, R=None, p=np.zeros(3), s=1.0, ax_s=1,
               strict_check=True, **kwargs):
    """Plot basis of a rotation matrix.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    R : array-like, shape (3, 3), optional (default: I)
        Rotation matrix, each column contains a basis vector

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the frame that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from ..plot_utils import make_3d_axis, Frame
    if ax is None:
        ax = make_3d_axis(ax_s)

    if R is None:
        R = np.eye(3)
    R = check_matrix(R, strict_check=strict_check)

    A2B = np.eye(4)
    A2B[:3, :3] = R
    A2B[:3, 3] = p

    frame = Frame(A2B, s=s, **kwargs)
    frame.add_frame(ax)

    return ax


def plot_axis_angle(ax=None, a=a_id, p=p0, s=1.0, ax_s=1, **kwargs):
    """Plot rotation axis and angle.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    a : array-like, shape (4,), optional (default: [1, 0, 0, 0])
        Axis of rotation and rotation angle: (x, y, z, angle)

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from ..plot_utils import make_3d_axis, Arrow3D
    a = check_axis_angle(a)
    if ax is None:
        ax = make_3d_axis(ax_s)

    axis_arrow = Arrow3D(
        [p[0], p[0] + s * a[0]],
        [p[1], p[1] + s * a[1]],
        [p[2], p[2] + s * a[2]],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(axis_arrow)

    p1 = (unitx if np.abs(a[0]) <= np.finfo(float).eps else
          perpendicular_to_vectors(unity, a[:3]))
    p2 = perpendicular_to_vectors(a[:3], p1)

    angle_p1p2 = angle_between_vectors(p1, p2)
    arc = np.empty((100, 3))
    for i, t in enumerate(np.linspace(0, 2 * a[3] / np.pi, len(arc))):
        w1, w2 = slerp_weights(angle_p1p2, t)
        arc[i] = p + 0.5 * s * (a[:3] + w1 * p1 + w2 * p2)
    ax.plot(arc[:-5, 0], arc[:-5, 1], arc[:-5, 2], color="k", lw=3, **kwargs)

    arrow_coords = np.vstack((arc[-1], arc[-1] + 20 * (arc[-1] - arc[-3]))).T
    angle_arrow = Arrow3D(
        arrow_coords[0], arrow_coords[1], arrow_coords[2],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(angle_arrow)

    for i in [0, -1]:
        arc_bound = np.vstack((p + 0.5 * s * a[:3], arc[i])).T
        ax.plot(arc_bound[0], arc_bound[1], arc_bound[2], "--", c="k")

    return ax


def plot_wedge(ax=None, a=None, b=None, ax_s=1, **kwargs):  # TODO type hints, sphinx
    """Plot bivector from wedge product of two vectors.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    a : TODO
    b : TODO

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from ..plot_utils import make_3d_axis, plot_vector
    if ax is None:
        ax = make_3d_axis(ax_s)

    if a is None:
        a = np.array([1, 0, 0])
    if b is None:
        b = np.array([0, 1, 0])

    _plot_parallelogram(ax, a, b, "#aaaaaa", 0.5)
    a_xy = np.array([a[0], a[1], 0])
    b_xy = np.array([b[0], b[1], 0])
    _plot_parallelogram(ax, a_xy, b_xy, "#ffff00", 0.5)
    a_xz = np.array([a[0], 0, a[2]])
    b_xz = np.array([b[0], 0, b[2]])
    _plot_parallelogram(ax, a_xz, b_xz, "#ff00ff", 0.5)
    a_yz = np.array([0, a[1], a[2]])
    b_yz = np.array([0, b[1], b[2]])
    _plot_parallelogram(ax, a_yz, b_yz, "#00ffff", 0.5)

    return ax


def _plot_parallelogram(ax, a, b, color, alpha):
    from ..plot_utils import plot_vector
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plot_vector(ax, start=np.zeros(3), direction=a, color=color, alpha=alpha)
    plot_vector(ax, start=np.zeros(3), direction=b, color=color, alpha=alpha)
    parallelogram = Poly3DCollection(np.vstack((np.zeros(3), a, b, a + b)))
    parallelogram.set_facecolor(color)
    parallelogram.set_alpha(alpha)
    ax.add_collection3d(parallelogram)
