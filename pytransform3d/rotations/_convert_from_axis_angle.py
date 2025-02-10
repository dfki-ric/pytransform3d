"""Conversions from axis-angle to other representations."""
import math
import numpy as np
from ._utils import check_axis_angle, check_compact_axis_angle


def matrix_from_axis_angle(a):
    r"""Compute rotation matrix from axis-angle.

    This is called exponential map or Rodrigues' formula.

    .. math::

        \boldsymbol{R}(\hat{\boldsymbol{\omega}}, \theta)
        =
        Exp(\hat{\boldsymbol{\omega}} \theta)
        =
        \cos{\theta} \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta})
        \hat{\boldsymbol{\omega}}\hat{\boldsymbol{\omega}}^T
        =
        \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta}) \left[\hat{\boldsymbol{\omega}}\right]^2

    This typically results in an active rotation matrix.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    a = check_axis_angle(a)
    ux, uy, uz, theta = a
    c = math.cos(theta)
    s = math.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(a[3]) +
    #      (1.0 - np.cos(a[3])) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(a[3]))
    # or
    # w = cross_product_matrix(a[:3])
    # R = np.eye(3) + np.sin(a[3]) * w + (1.0 - np.cos(a[3])) * w.dot(w)

    return R


def matrix_from_compact_axis_angle(a):
    r"""Compute rotation matrix from compact axis-angle.

    This is called exponential map or Rodrigues' formula.

    .. math::

        Exp(\hat{\boldsymbol{\omega}} \theta)
        =
        \cos{\theta} \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta})
        \hat{\boldsymbol{\omega}}\hat{\boldsymbol{\omega}}^T
        =
        \boldsymbol{I}
        + \sin{\theta} \left[\hat{\boldsymbol{\omega}}\right]
        + (1 - \cos{\theta}) \left[\hat{\boldsymbol{\omega}}\right]^2

    This typically results in an active rotation matrix.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    a = axis_angle_from_compact_axis_angle(a)
    return matrix_from_axis_angle(a)


def axis_angle_from_compact_axis_angle(a):
    """Compute axis-angle from compact axis-angle representation.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    a = check_compact_axis_angle(a)
    angle = np.linalg.norm(a)

    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = a / angle
    return np.hstack((axis, (angle,)))


def quaternion_from_axis_angle(a):
    """Compute quaternion from axis-angle.

    This operation is called exponential map.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    a = check_axis_angle(a)
    half_angle = 0.5 * a[3]

    q = np.empty(4)
    q[0] = np.cos(half_angle)
    q[1:] = np.sin(half_angle) * a[:3]
    return q


def quaternion_from_compact_axis_angle(a):
    """Compute quaternion from compact axis-angle (exponential map).

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    a = axis_angle_from_compact_axis_angle(a)
    return quaternion_from_axis_angle(a)
