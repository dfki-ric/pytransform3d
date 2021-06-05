"""Spherical linear interpolation (SLERP)."""
import numpy as np
from ._utils import (check_axis_angle, check_quaternion, angle_between_vectors,
                     check_rotor)


def axis_angle_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start axis of rotation and rotation angle: (x, y, z, angle)

    end : array-like, shape (4,)
        Goal axis of rotation and rotation angle: (x, y, z, angle)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    a : array, shape (4,)
        Interpolated axis of rotation and rotation angle: (x, y, z, angle)
    """
    start = check_axis_angle(start)
    end = check_axis_angle(end)
    angle = angle_between_vectors(start[:3], end[:3])
    w1, w2 = slerp_weights(angle, t)
    w1 = np.array([w1, w1, w1, (1.0 - t)])
    w2 = np.array([w2, w2, w2, t])
    return w1 * start + w2 * end


def quaternion_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start unit quaternion to represent rotation: (w, x, y, z)

    end : array-like, shape (4,)
        End unit quaternion to represent rotation: (w, x, y, z)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    q : array, shape (4,)
        Interpolated unit quaternion to represent rotation: (w, x, y, z)
    """
    start = check_quaternion(start)
    end = check_quaternion(end)
    angle = angle_between_vectors(start, end)
    w1, w2 = slerp_weights(angle, t)
    return w1 * start + w2 * end


def rotor_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    end : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    rotor : array, shape (4,)
        Interpolated rotor: (a, b_yz, b_zx, b_xy)
    """
    start = check_rotor(start)
    end = check_rotor(end)
    return quaternion_slerp(start, end, t)


def slerp_weights(angle, t):
    """Compute weights of start and end for spherical linear interpolation."""
    if angle == 0.0:
        return np.ones_like(t), np.zeros_like(t)
    else:
        return (np.sin((1.0 - t) * angle) / np.sin(angle),
                np.sin(t * angle) / np.sin(angle))
