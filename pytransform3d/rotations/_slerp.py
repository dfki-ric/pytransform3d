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
        Position between start and end

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


def quaternion_slerp(start, end, t, shortest_path=False):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start unit quaternion to represent rotation: (w, x, y, z)

    end : array-like, shape (4,)
        End unit quaternion to represent rotation: (w, x, y, z)

    t : float in [0, 1]
        Position between start and end

    shortest_path : bool, optional (default: False)
        Resolve sign ambiguity before interpolation to find the shortest path.
        The end quaternion will be picked to be close to the start quaternion.

    Returns
    -------
    q : array, shape (4,)
        Interpolated unit quaternion to represent rotation: (w, x, y, z)
    """
    start = check_quaternion(start)
    end = check_quaternion(end)
    if shortest_path:
        end = _pick_closest_quaternion(end, start)
    angle = angle_between_vectors(start, end)
    w1, w2 = slerp_weights(angle, t)
    return w1 * start + w2 * end


def pick_closest_quaternion(quaternion, target_quaternion):
    """Resolve quaternion ambiguity and pick the closest one to the target.

    .. warning::
        There are always two quaternions that represent the exact same
        orientation: q and -q.

    Parameters
    ----------
    quaternion : array-like, shape (4,)
        Quaternion (w, x, y, z) of which we are unsure whether we want to
        select quaternion or -quaternion.

    target_quaternion : array-like, shape (4,)
        Target quaternion (w, x, y, z) to which we want to be close.

    Returns
    -------
    closest_quaternion : array, shape (4,)
        Quaternion that is closest (Euclidean norm) to the target quaternion.
    """
    quaternion = check_quaternion(quaternion)
    target_quaternion = check_quaternion(target_quaternion)
    return _pick_closest_quaternion(quaternion, target_quaternion)


def _pick_closest_quaternion(quaternion, target_quaternion):
    """Resolve quaternion ambiguity and pick the closest one to the target.

    This is an internal function that does not validate the inputs.

    Parameters
    ----------
    quaternion : array, shape (4,)
        Quaternion (w, x, y, z) of which we are unsure whether we want to
        select quaternion or -quaternion.

    target_quaternion : array, shape (4,)
        Target quaternion (w, x, y, z) to which we want to be close.

    Returns
    -------
    closest_quaternion : array, shape (4,)
        Quaternion that is closest (Euclidean norm) to the target quaternion.
    """
    if (np.linalg.norm(-quaternion - target_quaternion) <
            np.linalg.norm(quaternion - target_quaternion)):
        return -quaternion
    return quaternion


def rotor_slerp(start, end, t, shortest_path=False):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    end : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    t : float in [0, 1]
        Position between start and end

    shortest_path : bool, optional (default: False)
        Resolve sign ambiguity before interpolation to find the shortest path.
        The end rotor will be picked to be close to the start rotor.

    Returns
    -------
    rotor : array, shape (4,)
        Interpolated rotor: (a, b_yz, b_zx, b_xy)
    """
    start = check_rotor(start)
    end = check_rotor(end)
    return quaternion_slerp(start, end, t, shortest_path)


def slerp_weights(angle, t):
    """Compute weights of start and end for spherical linear interpolation.

    Parameters
    ----------
    angle : float
        Rotation angle.

    t : float or array, shape (n_steps,)
        Position between start and end

    Returns
    -------
    w1 : float or array, shape (n_steps,)
        Weights for quaternion 1

    w2 : float or array, shape (n_steps,)
        Weights for quaternion 2
    """
    if angle == 0.0:
        return np.ones_like(t), np.zeros_like(t)
    return (np.sin((1.0 - t) * angle) / np.sin(angle),
            np.sin(t * angle) / np.sin(angle))
