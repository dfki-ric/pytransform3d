"""Plotting utilities."""
import numpy as np
from ._utils import check_transform, check_screw_parameters
from ._transform_operations import (
    transform, vector_to_point, vector_to_direction, vectors_to_points)


def plot_transform(ax=None, A2B=None, s=1.0, ax_s=1, name=None,
                   strict_check=True, **kwargs):
    """Plot transform.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    A2B : array-like, shape (4, 4), optional (default: I)
        Transform from frame A to frame B

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    name : string, optional (default: None)
        Name of the frame, will be used for annotation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

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

    if A2B is None:
        A2B = np.eye(4)
    A2B = check_transform(A2B, strict_check=strict_check)

    frame = Frame(A2B, name, s, **kwargs)
    frame.add_frame(ax)

    return ax


def plot_screw(ax=None, q=np.zeros(3), s_axis=np.array([1.0, 0.0, 0.0]), h=1.0,
               theta=1.0, A2B=None, s=1.0, ax_s=1, alpha=1.0, **kwargs):
    """Plot transformation about and along screw axis.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    q : array-like, shape (3,), optional (default: [0, 0, 0])
        Vector to a point on the screw axis

    s_axis : array-like, shape (3,), optional (default: [1, 0, 0])
        Direction vector of the screw axis

    h : float, optional (default: 1)
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta : float, optional (default: 1)
        Rotation angle. h * theta is the translation.

    A2B : array-like, shape (4, 4), optional (default: I)
        Origin of the screw

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    alpha : float, optional (default: 1)
        Alpha channel of plotted lines

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. color

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from ..plot_utils import make_3d_axis, Arrow3D
    from ..rotations import (vector_projection, angle_between_vectors,
                             perpendicular_to_vectors, slerp_weights)

    if ax is None:
        ax = make_3d_axis(ax_s)

    if A2B is None:
        A2B = np.eye(4)

    q, s_axis, h = check_screw_parameters(q, s_axis, h)

    origin_projected_on_screw_axis = q + vector_projection(-q, s_axis)

    pure_translation = np.isinf(h)

    if not pure_translation:
        screw_axis_to_old_frame = -origin_projected_on_screw_axis
        screw_axis_to_rotated_frame = perpendicular_to_vectors(
            s_axis, screw_axis_to_old_frame)
        screw_axis_to_translated_frame = h * s_axis

        arc = np.empty((100, 3))
        angle = angle_between_vectors(
            screw_axis_to_old_frame, screw_axis_to_rotated_frame)
        for i, t in enumerate(zip(np.linspace(0, 2 * theta / np.pi, len(arc)),
                                  np.linspace(0.0, 1.0, len(arc)))):
            t1, t2 = t
            w1, w2 = slerp_weights(angle, t1)
            arc[i] = (origin_projected_on_screw_axis
                      + w1 * screw_axis_to_old_frame
                      + w2 * screw_axis_to_rotated_frame
                      + screw_axis_to_translated_frame * t2 * theta)

    q = transform(A2B, vector_to_point(q))[:3]
    s_axis = transform(A2B, vector_to_direction(s_axis))[:3]
    if not pure_translation:
        arc = transform(A2B, vectors_to_points(arc))[:, :3]
        origin_projected_on_screw_axis = transform(
            A2B, vector_to_point(origin_projected_on_screw_axis))[:3]

    # Screw axis
    ax.scatter(q[0], q[1], q[2], color="r")
    if pure_translation:
        s_axis *= theta
        ax.scatter(q[0] + s_axis[0], q[1] + s_axis[1], q[2] + s_axis[2],
                   color="r")
    ax.plot(
        [q[0] - s * s_axis[0], q[0] + (1 + s) * s_axis[0]],
        [q[1] - s * s_axis[1], q[1] + (1 + s) * s_axis[1]],
        [q[2] - s * s_axis[2], q[2] + (1 + s) * s_axis[2]],
        "--", c="k", alpha=alpha)
    axis_arrow = Arrow3D(
        [q[0], q[0] + s_axis[0]],
        [q[1], q[1] + s_axis[1]],
        [q[2], q[2] + s_axis[2]],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k", alpha=alpha)
    ax.add_artist(axis_arrow)

    if not pure_translation:
        # Transformation
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", lw=3,
                alpha=alpha, **kwargs)
        arrow_coords = np.vstack((arc[-1], arc[-1] + (arc[-1] - arc[-2]))).T
        angle_arrow = Arrow3D(
            arrow_coords[0], arrow_coords[1], arrow_coords[2],
            mutation_scale=20, lw=3, arrowstyle="-|>", color="k", alpha=alpha)
        ax.add_artist(angle_arrow)

        for i in [0, -1]:
            arc_bound = np.vstack((origin_projected_on_screw_axis, arc[i])).T
            ax.plot(arc_bound[0], arc_bound[1], arc_bound[2], "--", c="k",
                    alpha=alpha)

    return ax
