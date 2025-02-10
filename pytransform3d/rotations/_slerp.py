"""Spherical linear interpolation (SLERP)."""
import numpy as np
from ._utils import angle_between_vectors
from ._matrix import compact_axis_angle_from_matrix
from ._axis_angle import check_axis_angle, matrix_from_compact_axis_angle
from ._quaternions import check_quaternion, pick_closest_quaternion_impl
from ._rotors import check_rotor


def matrix_slerp(start, end, t):
    r"""Spherical linear interpolation (SLERP) for rotation matrices.

    We compute the difference between two orientations
    :math:`\boldsymbol{R}_{AB} = \boldsymbol{R}^{-1}_{WA}\boldsymbol{R}_{WB}`,
    convert it to a rotation vector
    :math:`Log(\boldsymbol{R}_{AB}) = \boldsymbol{\omega}_{AB}`, compute a
    fraction of it :math:`t\boldsymbol{\omega}_{AB}` with :math:`t \in [0, 1]`,
    and use the exponential map to concatenate the fraction of the difference
    :math:`\boldsymbol{R}_{WA} Exp(t\boldsymbol{\omega}_{AB})`.

    Parameters
    ----------
    start : array-like, shape (3, 3)
        Rotation matrix to represent start orientation.

    end : array-like, shape (3, 3)
        Rotation matrix to represent end orientation.

    t : float in [0, 1]
        Position between start and goal.

    Returns
    -------
    R_t : array, shape (3, 3)
        Interpolated orientation.

    See Also
    --------
    axis_angle_slerp :
        SLERP axis-angle representation.

    quaternion_slerp :
        SLERP for quaternions.

    rotor_slerp :
        SLERP for rotors.

    pytransform3d.transformations.pq_slerp :
        SLERP for position + quaternion.
    """
    end2start = np.dot(np.transpose(start), end)
    return np.dot(start, matrix_power(end2start, t))


def matrix_power(R, t):
    r"""Compute power of a rotation matrix with respect to scalar.

    .. math::

        \boldsymbol{R}^t

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix.

    t : float
        Exponent.

    Returns
    -------
    R_t : array, shape (3, 3)
        Rotation matrix.
    """
    a = compact_axis_angle_from_matrix(R)
    return matrix_from_compact_axis_angle(a * t)


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

    See Also
    --------
    matrix_slerp :
        SLERP for rotation matrices.

    quaternion_slerp :
        SLERP for quaternions.

    rotor_slerp :
        SLERP for rotors.

    pytransform3d.transformations.pq_slerp :
        SLERP for position + quaternion.
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

    See Also
    --------
    matrix_slerp :
        SLERP for rotation matrices.

    axis_angle_slerp :
        SLERP for axis-angle representation.

    rotor_slerp :
        SLERP for rotors.

    pytransform3d.transformations.pq_slerp :
        SLERP for position + quaternion.
    """
    start = check_quaternion(start)
    end = check_quaternion(end)
    if shortest_path:
        end = pick_closest_quaternion_impl(end, start)
    angle = angle_between_vectors(start, end)
    w1, w2 = slerp_weights(angle, t)
    return w1 * start + w2 * end


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

    See Also
    --------
    matrix_slerp :
        SLERP for rotation matrices.

    axis_angle_slerp :
        SLERP for axis-angle representation.

    quaternion_slerp :
        SLERP for quaternions.

    pytransform3d.transformations.pq_slerp :
        SLERP for position + quaternion.
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
