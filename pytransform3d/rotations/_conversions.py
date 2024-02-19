"""Conversions between rotation representations."""
import math
import warnings
import numpy as np
from ._utils import (
    check_matrix, check_quaternion, check_axis_angle, check_compact_axis_angle,
    check_mrp, norm_angle, norm_vector, norm_axis_angle,
    perpendicular_to_vector, perpendicular_to_vectors, vector_projection)
from ._constants import unitx, unity, unitz, eps


def cross_product_matrix(v):
    r"""Generate the cross-product matrix of a vector.

    The cross-product matrix :math:`\boldsymbol{V}` satisfies the equation

    .. math::

        \boldsymbol{V} \boldsymbol{w} = \boldsymbol{v} \times
        \boldsymbol{w}.

    It is a skew-symmetric (antisymmetric) matrix, i.e.,
    :math:`-\boldsymbol{V} = \boldsymbol{V}^T`. Its elements are

    .. math::

        \left[\boldsymbol{v}\right]
        =
        \left[\begin{array}{c}
        v_1\\ v_2\\ v_3
        \end{array}\right]
        =
        \boldsymbol{V}
        =
        \left(\begin{array}{ccc}
        0 & -v_3 & v_2\\
        v_3 & 0 & -v_1\\
        -v_2 & v_1 & 0
        \end{array}\right).

    Parameters
    ----------
    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    V : array-like, shape (3, 3)
        Cross-product matrix
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def matrix_from_two_vectors(a, b):
    """Compute rotation matrix from two vectors.

    We assume that the two given vectors form a plane so that we can compute
    a third, orthogonal vector with the cross product.

    The x-axis will point in the same direction as a, the y-axis corresponds
    to the normalized vector rejection of b on a, and the z-axis is the
    cross product of the other basis vectors.

    Parameters
    ----------
    a : array-like, shape (3,)
        First vector, must not be 0

    b : array-like, shape (3,)
        Second vector, must not be 0 or parallel to a

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix

    Raises
    ------
    ValueError
        If vectors are parallel or one of them is the zero vector
    """
    if np.linalg.norm(a) == 0:
        raise ValueError("a must not be the zero vector.")
    if np.linalg.norm(b) == 0:
        raise ValueError("b must not be the zero vector.")

    c = perpendicular_to_vectors(a, b)
    if np.linalg.norm(c) == 0:
        raise ValueError("a and b must not be parallel.")

    a = norm_vector(a)

    b_on_a_projection = vector_projection(b, a)
    b_on_a_rejection = b - b_on_a_projection
    b = norm_vector(b_on_a_rejection)

    c = norm_vector(c)

    return np.column_stack((a, b, c))


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


def matrix_from_quaternion(q):
    """Compute rotation matrix from quaternion.

    This typically results in an active rotation matrix.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    q = check_quaternion(q)
    uq = norm_vector(q)
    w, x, y, z = uq
    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    R = np.array([[1.0 - y2 - z2, xy - zw, xz + yw],
                  [xy + zw, 1.0 - x2 - z2, yz - xw],
                  [xz - yw, yz + xw, 1.0 - x2 - y2]])
    return R


def passive_matrix_from_angle(basis, angle):
    """Compute passive rotation matrix from rotation about basis vector.

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angle : float
        Rotation angle

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix

    Raises
    ------
    ValueError
        If basis is invalid
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if basis == 0:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, c, s],
                      [0.0, -s, c]])
    elif basis == 1:
        R = np.array([[c, 0.0, -s],
                      [0.0, 1.0, 0.0],
                      [s, 0.0, c]])
    elif basis == 2:
        R = np.array([[c, s, 0.0],
                      [-s, c, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R


def active_matrix_from_angle(basis, angle):
    r"""Compute active rotation matrix from rotation about basis vector.

    With the angle :math:`\alpha` and :math:`s = \sin{\alpha}, c=\cos{\alpha}`,
    we construct rotation matrices about the basis vectors as follows:

    .. math::

        \boldsymbol{R}_x(\alpha) =
        \left(
        \begin{array}{ccc}
        1 & 0 & 0\\
        0 & c & -s\\
        0 & s & c
        \end{array}
        \right)

    .. math::

        \boldsymbol{R}_y(\alpha) =
        \left(
        \begin{array}{ccc}
        c & 0 & s\\
        0 & 1 & 0\\
        -s & 0 & c
        \end{array}
        \right)

    .. math::

        \boldsymbol{R}_z(\alpha) =
        \left(
        \begin{array}{ccc}
        c & -s & 0\\
        s & c & 0\\
        0 & 0 & 1
        \end{array}
        \right)

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angle : float
        Rotation angle

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix

    Raises
    ------
    ValueError
        If basis is invalid
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if basis == 0:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0, c, -s],
                      [0.0, s, c]])
    elif basis == 1:
        R = np.array([[c, 0.0, s],
                      [0.0, 1.0, 0.0],
                      [-s, 0.0, c]])
    elif basis == 2:
        R = np.array([[c, -s, 0.0],
                      [s, c, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R


def active_matrix_from_intrinsic_euler_xzx(e):
    """Compute active rotation matrix from intrinsic xzx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, alpha).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(0, gamma))
    return R


def active_matrix_from_extrinsic_euler_xzx(e):
    """Compute active rotation matrix from extrinsic xzx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, gamma).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(0, alpha))
    return R


def active_matrix_from_intrinsic_euler_xyx(e):
    """Compute active rotation matrix from intrinsic xyx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, alpha).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(0, gamma))
    return R


def active_matrix_from_extrinsic_euler_xyx(e):
    """Compute active rotation matrix from extrinsic xyx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, gamma).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(0, alpha))
    return R


def active_matrix_from_intrinsic_euler_yxy(e):
    """Compute active rotation matrix from intrinsic yxy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, alpha).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(1, gamma))
    return R


def active_matrix_from_extrinsic_euler_yxy(e):
    """Compute active rotation matrix from extrinsic yxy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, gamma).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(1, alpha))
    return R


def active_matrix_from_intrinsic_euler_yzy(e):
    """Compute active rotation matrix from intrinsic yzy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, alpha).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(1, gamma))
    return R


def active_matrix_from_extrinsic_euler_yzy(e):
    """Compute active rotation matrix from extrinsic yzy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, gamma).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(1, alpha))
    return R


def active_matrix_from_intrinsic_euler_zyz(e):
    """Compute active rotation matrix from intrinsic zyz Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, alpha).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(2, gamma))
    return R


def active_matrix_from_extrinsic_euler_zyz(e):
    """Compute active rotation matrix from extrinsic zyz Euler angles.

    .. warning::

        This function was not implemented correctly in versions 1.3 and 1.4
        as the order of the angles was reversed, which actually corresponds
        to intrinsic rotations. This has been fixed in version 1.5.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, gamma).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(2, alpha))
    return R


def active_matrix_from_intrinsic_euler_zxz(e):
    """Compute active rotation matrix from intrinsic zxz Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, alpha).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(2, gamma))
    return R


def active_matrix_from_extrinsic_euler_zxz(e):
    """Compute active rotation matrix from extrinsic zxz Euler angles.

    .. warning::

        This function was not implemented correctly in versions 1.3 and 1.4
        as the order of the angles was reversed, which actually corresponds
        to intrinsic rotations. This has been fixed in version 1.5.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, gamma).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(2, alpha))
    return R


def active_matrix_from_intrinsic_euler_xzy(e):
    """Compute active rotation matrix from intrinsic xzy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, alpha).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(1, gamma))
    return R


def active_matrix_from_extrinsic_euler_xzy(e):
    """Compute active rotation matrix from extrinsic xzy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, gamma).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(0, alpha))
    return R


def active_matrix_from_intrinsic_euler_xyz(e):
    """Compute active rotation matrix from intrinsic xyz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, alpha).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(2, gamma))
    return R


def active_matrix_from_extrinsic_euler_xyz(e):
    """Compute active rotation matrix from extrinsic xyz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, gamma).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(0, alpha))
    return R


def active_matrix_from_intrinsic_euler_yxz(e):
    """Compute active rotation matrix from intrinsic yxz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, alpha).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(2, gamma))
    return R


def active_matrix_from_extrinsic_euler_yxz(e):
    """Compute active rotation matrix from extrinsic yxz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, gamma).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(1, alpha))
    return R


def active_matrix_from_intrinsic_euler_yzx(e):
    """Compute active rotation matrix from intrinsic yzx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, alpha).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(0, gamma))
    return R


def active_matrix_from_extrinsic_euler_yzx(e):
    """Compute active rotation matrix from extrinsic yzx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, gamma).dot(
        active_matrix_from_angle(2, beta)).dot(
        active_matrix_from_angle(1, alpha))
    return R


def active_matrix_from_intrinsic_euler_zyx(e):
    """Compute active rotation matrix from intrinsic zyx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, alpha).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(0, gamma))
    return R


def active_matrix_from_extrinsic_euler_zyx(e):
    """Compute active rotation matrix from extrinsic zyx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(0, gamma).dot(
        active_matrix_from_angle(1, beta)).dot(
        active_matrix_from_angle(2, alpha))
    return R


def active_matrix_from_intrinsic_euler_zxy(e):
    """Compute active rotation matrix from intrinsic zxy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, alpha).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(1, gamma))
    return R


def active_matrix_from_extrinsic_euler_zxy(e):
    """Compute active rotation matrix from extrinsic zxy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(1, gamma).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(2, alpha))
    return R


def active_matrix_from_extrinsic_roll_pitch_yaw(rpy):
    """Compute active rotation matrix from extrinsic roll, pitch, and yaw.

    Parameters
    ----------
    rpy : array-like, shape (3,)
        Angles for rotation around x- (roll), y- (pitch), and z-axes (yaw),
        extrinsic rotations

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    return active_matrix_from_extrinsic_euler_xyz(rpy)


def check_axis_index(name, i):
    """Checks axis index.

    Parameters
    ----------
    name : str
        Name of the axis. Required for the error message.

    i : int from [0, 1, 2]
        Index of the axis (0: x, 1: y, 2: z)

    Raises
    ------
    ValueError
        If basis is invalid
    """
    if i not in [0, 1, 2]:
        raise ValueError("Axis index %s (%d) must be in [0, 1, 2]" % (name, i))


def matrix_from_euler(e, i, j, k, extrinsic):
    """General method to compute active rotation matrix from any Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Extracted rotation angles in radians about the axes i, j, k in this
        order. The first and last angle are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    Returns
    -------
    R : array, shape (3, 3)
        Active rotation matrix

    Raises
    ------
    ValueError
        If basis is invalid
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    alpha, beta, gamma = e
    if not extrinsic:
        i, k = k, i
        alpha, gamma = gamma, alpha
    R = active_matrix_from_angle(k, gamma).dot(
        active_matrix_from_angle(j, beta)).dot(
        active_matrix_from_angle(i, alpha))
    return R


def _general_intrinsic_euler_from_active_matrix(
        R, n1, n2, n3, proper_euler, strict_check=True):
    """General algorithm to extract intrinsic euler angles from a matrix.

    The implementation is based on SciPy's implementation:
    https://github.com/scipy/scipy/blob/743c283bbe79473a03ca2eddaa537661846d8a19/scipy/spatial/transform/_rotation.pyx

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Active rotation matrix

    n1 : array, shape (3,)
        First rotation axis (basis vector)

    n2 : array, shape (3,)
        Second rotation axis (basis vector)

    n3 : array, shape (3,)
        Third rotation axis (basis vector)

    proper_euler : bool
        Is this an Euler angle convention or a Cardan / Tait-Bryan convention?
        Proper Euler angles rotate about the same axis twice, for example,
        z, y', and z''.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    euler_angles : array, shape (3,)
        Extracted intrinsic rotation angles in radians about the axes
        n1, n2, and n3 in this order. The first and last angle are
        normalized to [-pi, pi]. The middle angle is normalized to
        either [0, pi] (proper Euler angles) or [-pi/2, pi/2]
        (Cardan / Tait-Bryan angles).

    References
    ----------
    .. [1] Shuster, M. D., Markley, F. L. (2006).
       General Formula for Extracting the Euler Angles.
       Journal of Guidance, Control, and Dynamics, 29(1), pp 2015-221,
       doi: 10.2514/1.16622. https://arc.aiaa.org/doi/abs/10.2514/1.16622
    """
    D = check_matrix(R, strict_check=strict_check)

    # Differences to the paper:
    # - we call the angles alpha, beta, and gamma
    # - we obtain angles from intrinsic rotations, thus some matrices are
    #   transposed like in SciPy's implementation

    # Step 2
    # - Equation 5
    n1_cross_n2 = np.cross(n1, n2)
    lmbda = np.arctan2(
        np.dot(n1_cross_n2, n3),
        np.dot(n1, n3)
    )
    # - Equation 6
    C = np.vstack((n2, n1_cross_n2, n1))

    # Step 3
    # - Equation 8
    CDCT = np.dot(np.dot(C, D), C.T)
    O = np.dot(CDCT, active_matrix_from_angle(0, lmbda).T)

    # Step 4
    # Fix numerical issue if O_22 is slightly out of range of arccos
    O_22 = max(min(O[2, 2], 1.0), -1.0)
    # - Equation 10a
    beta = lmbda + np.arccos(O_22)

    safe1 = abs(beta - lmbda) >= np.finfo(float).eps
    safe2 = abs(beta - lmbda - np.pi) >= np.finfo(float).eps
    if safe1 and safe2:  # Default case, no gimbal lock
        # Step 5
        # - Equation 10b
        alpha = np.arctan2(O[0, 2], -O[1, 2])
        # - Equation 10c
        gamma = np.arctan2(O[2, 0], O[2, 1])

        # Step 7
        if proper_euler:
            valid_beta = 0.0 <= beta <= np.pi
        else:  # Cardan / Tait-Bryan angles
            valid_beta = -0.5 * np.pi <= beta <= 0.5 * np.pi
        # - Equation 12
        if not valid_beta:
            alpha += np.pi
            beta = 2.0 * lmbda - beta
            gamma -= np.pi
    else:
        # Step 6 - Handle gimbal locks
        # a)
        gamma = 0.0
        if not safe1:
            # b)
            alpha = np.arctan2(O[1, 0] - O[0, 1], O[0, 0] + O[1, 1])
        else:
            # c)
            alpha = np.arctan2(O[1, 0] + O[0, 1], O[0, 0] - O[1, 1])
    euler_angles = norm_angle([alpha, beta, gamma])
    return euler_angles


def intrinsic_euler_xzx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xzx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z'-, and x''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unitx, True, strict_check)


def extrinsic_euler_xzx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xzx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z-, and x-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unitx, True, strict_check)[::-1]


def intrinsic_euler_xyx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xyx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y'-, and x''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitx, True, strict_check)


def extrinsic_euler_xyx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xyx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y-, and x-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitx, True, strict_check)[::-1]


def intrinsic_euler_yxy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yxy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x'-, and y''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unity, True, strict_check)


def extrinsic_euler_yxy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yxy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x-, and y-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unity, True, strict_check)[::-1]


def intrinsic_euler_yzy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yzy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z'-, and y''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unity, True, strict_check)


def extrinsic_euler_yzy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yzy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z-, and y-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unity, True, strict_check)[::-1]


def intrinsic_euler_zyz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zyz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y'-, and z''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitz, True, strict_check)


def extrinsic_euler_zyz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zyz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y-, and z-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitz, True, strict_check)[::-1]


def intrinsic_euler_zxz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zxz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x'-, and z''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unitz, True, strict_check)


def extrinsic_euler_zxz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zxz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x-, and z-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unitz, True, strict_check)[::-1]


def intrinsic_euler_xzy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xzy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z'-, and y''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unity, False, strict_check)


def extrinsic_euler_xzy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xzy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z-, and y-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unitx, False, strict_check)[::-1]


def intrinsic_euler_xyz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xyz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitz, False, strict_check)


def extrinsic_euler_xyz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xyz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitx, False, strict_check)[::-1]


def intrinsic_euler_yxz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yxz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x'-, and z''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unitz, False, strict_check)


def extrinsic_euler_yxz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yxz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x-, and z-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unity, False, strict_check)[::-1]


def intrinsic_euler_yzx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yzx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z'-, and x''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unitx, False, strict_check)


def extrinsic_euler_yzx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yzx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z-, and x-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unity, False, strict_check)[::-1]


def intrinsic_euler_zyx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zyx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitx, False, strict_check)


def extrinsic_euler_zyx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zyx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y-, and x-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitz, False, strict_check)[::-1]


def intrinsic_euler_zxy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zxy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x'-, and y''-axes (intrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unity, False, strict_check)


def extrinsic_euler_zxy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zxy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x-, and y-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unitz, False, strict_check)[::-1]


def euler_from_matrix(R, i, j, k, extrinsic, strict_check=True):
    """General method to extract any Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Active rotation matrix

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    euler_angles : array, shape (3,)
        Extracted rotation angles in radians about the axes i, j, k in this
        order. The first and last angle are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).

    Raises
    ------
    ValueError
        If basis is invalid

    References
    ----------
    .. [1] Shuster, M. D., Markley, F. L. (2006).
       General Formula for Extracting the Euler Angles.
       Journal of Guidance, Control, and Dynamics, 29(1), pp 2015-221,
       doi: 10.2514/1.16622. https://arc.aiaa.org/doi/abs/10.2514/1.16622
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    basis_vectors = [unitx, unity, unitz]
    proper_euler = i == k
    if extrinsic:
        i, k = k, i
    e = _general_intrinsic_euler_from_active_matrix(
        R, basis_vectors[i], basis_vectors[j], basis_vectors[k], proper_euler,
        strict_check)

    if extrinsic:
        e = e[::-1]

    return e


def euler_from_quaternion(q, i, j, k, extrinsic):
    """General method to extract any Euler angles from quaternions.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    Returns
    -------
    euler_angles : array, shape (3,)
        Extracted rotation angles in radians about the axes i, j, k in this
        order. The first and last angle are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).

    Raises
    ------
    ValueError
        If basis is invalid

    References
    ----------
    .. [1] Bernardes, E., Viollet, S. (2022). Quaternion to Euler angles
       conversion: A direct, general and computationally efficient method.
       PLOS ONE, 17(11), doi: 10.1371/journal.pone.0276302.
    """
    q = check_quaternion(q)

    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    i += 1
    j += 1
    k += 1

    # The original algorithm assumes extrinsic convention. Hence, we swap
    # the order of axes for intrinsic rotation.
    if not extrinsic:
        i, k = k, i

    # Proper Euler angles rotate about the same axis in the first and last
    # rotation. If this is not the case, they are called Cardan or Tait-Bryan
    # angles and have to be handled differently.
    proper_euler = i == k
    if proper_euler:
        k = 6 - i - j

    sign = (i - j) * (j - k) * (k - i) // 2
    a = q[0]
    b = q[i]
    c = q[j]
    d = q[k] * sign

    if not proper_euler:
        a, b, c, d = a - c, b + d, c + a, d - b

    # Equation 34 is used instead of Equation 35 as atan2 it is numerically
    # more accurate than acos.
    angle_j = 2.0 * math.atan2(math.hypot(c, d),
                               math.hypot(a, b))

    # Check for singularities
    if abs(angle_j) <= eps:
        singularity = 1
    elif abs(angle_j - math.pi) <= eps:
        singularity = 2
    else:
        singularity = 0

    # Equation 25
    # (theta_1 + theta_3) / 2
    half_sum = math.atan2(b, a)
    # (theta_1 - theta_3) / 2
    half_diff = math.atan2(d, c)

    if singularity == 0:  # no singularity
        # Equation 32
        angle_i = half_sum + half_diff
        angle_k = half_sum - half_diff
    elif extrinsic:  # singularity
        angle_k = 0.0
        if singularity == 1:
            angle_i = 2.0 * half_sum
        else:
            assert singularity == 2
            angle_i = 2.0 * half_diff
    else:  # intrinsic, singularity
        angle_i = 0.0
        if singularity == 1:
            angle_k = 2.0 * half_sum
        else:
            assert singularity == 2
            angle_k = -2.0 * half_diff

    if not proper_euler:
        # Equation 43
        angle_j -= math.pi / 2.0
        # Equation 44
        angle_i *= sign

    angle_k = norm_angle(angle_k)
    angle_i = norm_angle(angle_i)

    if extrinsic:
        return np.array([angle_k, angle_j, angle_i])

    return np.array([angle_i, angle_j, angle_k])


def axis_angle_from_matrix(R, strict_check=True, check=True):
    """Compute axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    We usually assume active rotations.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    check : bool, optional (default: True)
        Check if rotation matrix is valid

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    if check:
        R = check_matrix(R, strict_check=strict_check)
    cos_angle = (np.trace(R) - 1.0) / 2.0
    angle = np.arccos(min(max(-1.0, cos_angle), 1.0))

    if angle == 0.0:  # R == np.eye(3)
        return np.array([1.0, 0.0, 0.0, 0.0])

    a = np.empty(4)

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if abs(angle - np.pi) < 1e-4:  # np.trace(R) close to -1
        # The threshold 1e-4 is a result from this discussion:
        # https://github.com/dfki-ric/pytransform3d/issues/43
        # The standard formula becomes numerically unstable, however,
        # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
        # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
        # squared values of the rotation axis on the diagonal of this matrix.
        # We can still use the original formula to reconstruct the signs of
        # the rotation axis correctly.

        # In case of floating point inaccuracies:
        R_diag = np.clip(np.diag(R), -1.0, 1.0)

        eeT_diag = 0.5 * (R_diag + 1.0)
        signs = np.sign(axis_unnormalized)
        signs[signs == 0.0] = 1.0
        a[:3] = np.sqrt(eeT_diag) * signs
    else:
        a[:3] = axis_unnormalized
        # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
        # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
        # but the following is much more precise for angles close to 0 or pi:
    a[:3] /= np.linalg.norm(a[:3])

    a[3] = angle
    return a


def axis_angle_from_quaternion(q):
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    We usually assume active rotations.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    q = check_quaternion(q)
    p = q[1:]
    p_norm = np.linalg.norm(p)

    if p_norm < np.finfo(float).eps:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = p / p_norm
    w_clamped = max(min(q[0], 1.0), -1.0)
    angle = (2.0 * np.arccos(w_clamped),)
    return norm_axis_angle(np.hstack((axis, angle)))


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


def axis_angle_from_two_directions(a, b):
    """Compute axis-angle representation from two direction vectors.

    The rotation will transform direction vector a to direction vector b.
    The direction vectors don't have to be normalized as this will be
    done internally. Note that there is more than one possible solution.

    Parameters
    ----------
    a : array-like, shape (3,)
        First direction vector

    b : array-like, shape (3,)
        Second direction vector

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    a = norm_vector(a)
    b = norm_vector(b)
    cos_angle = a.dot(b)
    if abs(-1.0 - cos_angle) < eps:
        # For 180 degree rotations we have an infinite number of solutions,
        # but we have to pick one axis.
        axis = perpendicular_to_vector(a)
    else:
        axis = np.cross(a, b)
    aa = np.empty(4)
    aa[:3] = norm_vector(axis)
    aa[3] = np.arccos(max(min(cos_angle, 1.0), -1.0))
    return norm_axis_angle(aa)


def compact_axis_angle(a):
    r"""Compute 3-dimensional axis-angle from a 4-dimensional one.

    In a 3-dimensional axis-angle, the 4th dimension (the rotation) is
    represented by the norm of the rotation axis vector, which means we
    transform :math:`\left( \boldsymbol{\hat{e}}, \theta \right)` to
    :math:`\theta \boldsymbol{\hat{e}}`.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle).

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z) (compact
        representation).
    """
    a = check_axis_angle(a)
    return a[:3] * a[3]


def compact_axis_angle_from_matrix(R, check=True):
    """Compute compact axis-angle from rotation matrix.

    This operation is called logarithmic map. Note that there are two possible
    solutions for the rotation axis when the angle is 180 degrees (pi).

    We usually assume active rotations.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    check : bool, optional (default: True)
        Check if rotation matrix is valid

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].
    """
    a = axis_angle_from_matrix(R, check=check)
    return compact_axis_angle(a)


def compact_axis_angle_from_quaternion(q):
    """Compute compact axis-angle from quaternion (logarithmic map).

    We usually assume active rotations.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].
    """
    a = axis_angle_from_quaternion(q)
    return compact_axis_angle(a)


def quaternion_from_matrix(R, strict_check=True):
    """Compute quaternion from rotation matrix.

    We usually assume active rotations.

    .. warning::

        When computing a quaternion from the rotation matrix there is a sign
        ambiguity: q and -q represent the same rotation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    R = check_matrix(R, strict_check=strict_check)
    q = np.empty(4)

    # Source:
    # http://www.euclideanspace.com/maths/geometry/rotations/conversions
    trace = np.trace(R)
    if trace > 0.0:
        sqrt_trace = np.sqrt(1.0 + trace)
        q[0] = 0.5 * sqrt_trace
        q[1] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
        q[2] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
        q[3] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            sqrt_trace = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = 0.5 / sqrt_trace * (R[2, 1] - R[1, 2])
            q[1] = 0.5 * sqrt_trace
            q[2] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[3] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
        elif R[1, 1] > R[2, 2]:
            sqrt_trace = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = 0.5 / sqrt_trace * (R[0, 2] - R[2, 0])
            q[1] = 0.5 / sqrt_trace * (R[1, 0] + R[0, 1])
            q[2] = 0.5 * sqrt_trace
            q[3] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
        else:
            sqrt_trace = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = 0.5 / sqrt_trace * (R[1, 0] - R[0, 1])
            q[1] = 0.5 / sqrt_trace * (R[0, 2] + R[2, 0])
            q[2] = 0.5 / sqrt_trace * (R[2, 1] + R[1, 2])
            q[3] = 0.5 * sqrt_trace
    return q


def quaternion_from_angle(basis, angle):
    """Compute quaternion from rotation about basis vector.

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angle : float
        Rotation angle

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Raises
    ------
    ValueError
        If basis is invalid
    """
    half_angle = 0.5 * angle
    c = math.cos(half_angle)
    s = math.sin(half_angle)

    if basis == 0:
        q = np.array([c, s, 0.0, 0.0])
    elif basis == 1:
        q = np.array([c, 0.0, s, 0.0])
    elif basis == 2:
        q = np.array([c, 0.0, 0.0, s])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return q


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
    theta = a[3]

    q = np.empty(4)
    q[0] = np.cos(theta / 2)
    q[1:] = np.sin(theta / 2) * a[:3]
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


def quaternion_xyzw_from_wxyz(q_wxyz):
    """Converts from w, x, y, z to x, y, z, w convention.

    Parameters
    ----------
    q_wxyz : array-like, shape (4,)
        Quaternion with scalar part before vector part

    Returns
    -------
    q_xyzw : array, shape (4,)
        Quaternion with scalar part after vector part
    """
    q_wxyz = check_quaternion(q_wxyz)
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])


def quaternion_wxyz_from_xyzw(q_xyzw):
    """Converts from x, y, z, w to w, x, y, z convention.

    Parameters
    ----------
    q_xyzw : array-like, shape (4,)
        Quaternion with scalar part after vector part

    Returns
    -------
    q_wxyz : array, shape (4,)
        Quaternion with scalar part before vector part
    """
    q_xyzw = check_quaternion(q_xyzw)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quaternion_from_extrinsic_euler_xyz(e):
    """Compute quaternion from extrinsic xyz Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    warnings.warn(
        "quaternion_from_extrinsic_euler_xyz is deprecated, use "
        "quaternion_from_euler", DeprecationWarning)
    R = active_matrix_from_extrinsic_euler_xyz(e)
    return quaternion_from_matrix(R)


def quaternion_from_mrp(mrp):
    """Compute quaternion from modified Rodrigues parameters.

    Parameters
    ----------
    mrp : array-like, shape (3,)
        Modified Rodrigues parameters.

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    mrp = check_mrp(mrp)
    dot_product_p1 = np.dot(mrp, mrp) + 1.0
    q = np.empty(4, dtype=float)
    q[0] = (2.0 - dot_product_p1) / dot_product_p1
    q[1:] = 2.0 * mrp / dot_product_p1
    return q


def mrp_from_quaternion(q):
    """Compute modified Rodrigues parameters from quaternion.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    mrp : array, shape (3,)
        Modified Rodrigues parameters.
    """
    q = check_quaternion(q)
    if q[0] < 0.0:
        q = -q
    return q[1:] / (1.0 + q[0])
