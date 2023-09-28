"""Conversions between transform representations."""
import numpy as np
from ._utils import (check_transform, check_pq, check_screw_axis,
                     check_screw_parameters, check_exponential_coordinates,
                     check_screw_matrix, check_transform_log,
                     check_dual_quaternion)
from ..rotations import (
    matrix_from_quaternion, quaternion_from_matrix,
    compact_axis_angle_from_matrix, matrix_from_compact_axis_angle,
    cross_product_matrix, q_conj, concatenate_quaternions,
    axis_angle_from_quaternion, norm_angle, eps)
from ..rotations import left_jacobian_SO3, left_jacobian_SO3_inv


def transform_from(R, p, strict_check=True):
    r"""Make transformation from rotation matrix and translation.

    .. math::

        \boldsymbol{T}_{BA} = \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right) \in SE(3)

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = rotate_transform(
        np.eye(4), R, strict_check=strict_check, check=False)
    A2B = translate_transform(
        A2B, p, strict_check=strict_check, check=False)
    return A2B


def translate_transform(A2B, p, strict_check=True, check=True):
    """Sets the translation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, -1] = p
    return out


def rotate_transform(A2B, R, strict_check=True, check=True):
    """Sets the rotation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, :3] = R
    return out


def pq_from_transform(A2B, strict_check=True):
    """Compute position and quaternion from transformation matrix.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transformation matrix from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    pq : array, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    return np.hstack((A2B[:3, 3], quaternion_from_matrix(A2B[:3, :3])))


def transform_from_pq(pq):
    """Compute transformation matrix from position and quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    A2B : array, shape (4, 4)
        Transformation matrix from frame A to frame B
    """
    pq = check_pq(pq)
    return transform_from(matrix_from_quaternion(pq[3:]), pq[:3])


def screw_parameters_from_screw_axis(screw_axis):
    """Compute screw parameters from screw axis.

    Note that there is not just one solution since q can be any point on the
    screw axis. We select q so that it is orthogonal to s_axis.

    Parameters
    ----------
    screw_axis : array-like, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    Returns
    -------
    q : array, shape (3,)
        Vector to a point on the screw axis that is orthogonal to s_axis

    s_axis : array, shape (3,)
        Unit direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.
    """
    screw_axis = check_screw_axis(screw_axis)

    omega = screw_axis[:3]
    v = screw_axis[3:]

    omega_norm = np.linalg.norm(omega)
    if abs(omega_norm) < np.finfo(float).eps:  # pure translation
        q = np.zeros(3)
        s_axis = v
        h = np.inf
    else:
        s_axis = omega
        h = omega.dot(v)
        moment = v - h * s_axis
        q = np.cross(s_axis, moment)
    return q, s_axis, h


def screw_axis_from_screw_parameters(q, s_axis, h):
    """Compute screw axis representation from screw parameters.

    Parameters
    ----------
    q : array-like, shape (3,)
        Vector to a point on the screw axis

    s_axis : array-like, shape (3,)
        Direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    Returns
    -------
    screw_axis : array, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.
    """
    q, s_axis, h = check_screw_parameters(q, s_axis, h)

    if np.isinf(h):  # pure translation
        return np.r_[0.0, 0.0, 0.0, s_axis]
    return np.r_[s_axis, np.cross(q, s_axis) + h * s_axis]


def screw_axis_from_exponential_coordinates(Stheta):
    """Compute screw axis and theta from exponential coordinates.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation. Theta should be >= 0. Negative rotations
        will be represented by a negative screw axis instead. This is relevant
        if you want to recover theta from exponential coordinates.

    Returns
    -------
    screw_axis : array, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    theta : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.
    """
    Stheta = check_exponential_coordinates(Stheta)

    omega_theta = Stheta[:3]
    v_theta = Stheta[3:]
    theta = np.linalg.norm(omega_theta)
    if theta < np.finfo(float).eps:
        theta = np.linalg.norm(v_theta)
    if theta < np.finfo(float).eps:
        return np.zeros(6), 0.0
    return Stheta / theta, theta


def screw_axis_from_screw_matrix(screw_matrix):
    """Compute screw axis from screw matrix.

    Parameters
    ----------
    screw_matrix : array-like, shape (4, 4)
        A screw matrix consists of a cross-product matrix that represents an
        axis of rotation, a translation, and a row of zeros.

    Returns
    -------
    screw_axis : array, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.
    """
    screw_matrix = check_screw_matrix(screw_matrix)

    screw_axis = np.empty(6)
    screw_axis[0] = screw_matrix[2, 1]
    screw_axis[1] = screw_matrix[0, 2]
    screw_axis[2] = screw_matrix[1, 0]
    screw_axis[3:] = screw_matrix[:3, 3]
    return screw_axis


def exponential_coordinates_from_screw_axis(screw_axis, theta):
    """Compute exponential coordinates from screw axis and theta.

    Parameters
    ----------
    screw_axis : array-like, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    theta : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    Returns
    -------
    Stheta : array, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation. Theta should be >= 0. Negative rotations
        will be represented by a negative screw axis instead. This is relevant
        if you want to recover theta from exponential coordinates.
    """
    screw_axis = check_screw_axis(screw_axis)
    return screw_axis * theta


def exponential_coordinates_from_transform_log(transform_log, check=True):
    r"""Compute exponential coordinates from logarithm of transformation.

    Extracts the vector :math:`\mathcal{S} \theta =
    (\hat{\boldsymbol{\omega}}, \boldsymbol{v}) \theta \in \mathbb{R}^6` from
    the matrix

    .. math::

        \left(
        \begin{array}{cccc}
        0 & -\omega_3 & \omega_2 & v_1\\
        \omega_3 & 0 & -\omega_1 & v_2\\
        -\omega_2 & \omega_1 & 0 & v_3\\
        0 & 0 & 0 & 0
        \end{array}
        \right)
        \theta = \left[ \mathcal{S} \right] \theta \in so(3).

    Parameters
    ----------
    transform_log : array-like, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.

    check : bool, optional (default: True)
        Check if logarithm of transformation is valid

    Returns
    -------
    Stheta : array, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.
    """
    if check:
        transform_log = check_transform_log(transform_log)

    Stheta = np.empty(6)
    Stheta[0] = transform_log[2, 1]
    Stheta[1] = transform_log[0, 2]
    Stheta[2] = transform_log[1, 0]
    Stheta[3:] = transform_log[:3, 3]
    return Stheta


def exponential_coordinates_from_transform(A2B, strict_check=True, check=True):
    r"""Compute exponential coordinates from transformation matrix.

    Logarithmic map.

    .. math::

        Log: \boldsymbol{T} \in SE(3)
        \rightarrow \mathcal{S} \theta \in \mathbb{R}^6

    .. math::

        Log(\boldsymbol{T}) =
        Log\left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)
        =
        \left(
        \begin{array}{c}
        Log(\boldsymbol{R})\\
        \boldsymbol{J}^{-1}(\theta) \boldsymbol{p}
        \end{array}
        \right)
        =
        \left(
        \begin{array}{c}
        \hat{\boldsymbol{\omega}}\\
        \boldsymbol{v}
        \end{array}
        \right)
        \theta
        =
        \mathcal{S}\theta,

    where :math:`\boldsymbol{J}^{-1}(\theta)` is the inverse left Jacobian of
    :math:`SO(3)` (see :func:`~pytransform3d.rotations.left_jacobian_SO3_inv`).

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transformation matrix from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    Stheta : array, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    if np.linalg.norm(np.eye(3) - R) < np.finfo(float).eps:
        return np.r_[0.0, 0.0, 0.0, p]

    omega_theta = compact_axis_angle_from_matrix(R, check=check)
    theta = np.linalg.norm(omega_theta)

    if theta == 0:
        return np.r_[0.0, 0.0, 0.0, p]

    v_theta = np.dot(left_jacobian_SO3_inv(omega_theta), p)

    return np.hstack((omega_theta, v_theta))


def screw_matrix_from_screw_axis(screw_axis):
    """Compute screw matrix from screw axis.

    Parameters
    ----------
    screw_axis : array-like, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    Returns
    -------
    screw_matrix : array, shape (4, 4)
        A screw matrix consists of a cross-product matrix that represents an
        axis of rotation, a translation, and a row of zeros.
    """
    screw_axis = check_screw_axis(screw_axis)

    omega = screw_axis[:3]
    v = screw_axis[3:]
    screw_matrix = np.zeros((4, 4))
    screw_matrix[:3, :3] = cross_product_matrix(omega)
    screw_matrix[:3, 3] = v
    return screw_matrix


def screw_matrix_from_transform_log(transform_log):
    """Compute screw matrix from logarithm of transformation.

    Parameters
    ----------
    transform_log : array-like, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.

    Returns
    -------
    screw_matrix : array, shape (4, 4)
        A screw matrix consists of a cross-product matrix that represents an
        axis of rotation, a translation, and a row of zeros.
    """
    transform_log = check_transform_log(transform_log)

    omega = np.array([
        transform_log[2, 1], transform_log[0, 2], transform_log[1, 0]])
    theta = np.linalg.norm(omega)
    if abs(theta) < np.finfo(float).eps:
        theta = np.linalg.norm(transform_log[:3, 3])
    if abs(theta) < np.finfo(float).eps:
        return np.zeros((4, 4)), 0.0
    return transform_log / theta, theta


def transform_log_from_exponential_coordinates(Stheta):
    r"""Compute matrix logarithm of transformation from exponential coordinates.

    Builds the matrix

    .. math::

        \left(
        \begin{array}{cccc}
        0 & -\omega_3 & \omega_2 & v_1\\
        \omega_3 & 0 & -\omega_1 & v_2\\
        -\omega_2 & \omega_1 & 0 & v_3\\
        0 & 0 & 0 & 0
        \end{array}
        \right) \theta
        = \left[ \mathcal{S} \right] \theta \in so(3)

    from the vector :math:`\mathcal{S} \theta = (\hat{\boldsymbol{\omega}},
    \boldsymbol{v}) \theta \in \mathbb{R}^6`.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    Returns
    -------
    transform_log : array, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.
    """
    check_exponential_coordinates(Stheta)

    omega = Stheta[:3]
    v = Stheta[3:]
    transform_log = np.zeros((4, 4))
    transform_log[:3, :3] = cross_product_matrix(omega)
    transform_log[:3, 3] = v
    return transform_log


def transform_log_from_screw_matrix(screw_matrix, theta):
    """Compute matrix logarithm of transformation from screw matrix and theta.

    Parameters
    ----------
    screw_matrix : array-like, shape (4, 4)
        A screw matrix consists of a cross-product matrix that represents an
        axis of rotation, a translation, and a row of zeros.

    theta : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    Returns
    -------
    transform_log : array, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.
    """
    screw_matrix = check_screw_matrix(screw_matrix)
    return screw_matrix * theta


def transform_log_from_transform(A2B, strict_check=True):
    r"""Compute matrix logarithm of transformation from transformation.

    Logarithmic map.

    .. math::

        \log: \boldsymbol{T} \in SE(3)
        \rightarrow \left[ \mathcal{S} \right] \theta \in se(3)

    .. math::

        \log(\boldsymbol{T}) =
        \log\left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)
        =
        \left(
        \begin{array}{cc}
        \log\boldsymbol{R} & \boldsymbol{J}^{-1}(\theta) \boldsymbol{p}\\
        \boldsymbol{0} & 0
        \end{array}
        \right)
        =
        \left(
        \begin{array}{cc}
        \hat{\boldsymbol{\omega}} \theta
        & \boldsymbol{v} \theta\\
        \boldsymbol{0} & 0
        \end{array}
        \right)
        =
        \left[\mathcal{S}\right]\theta,

    where :math:`\boldsymbol{J}^{-1}(\theta)` is the inverse left Jacobian of
    :math:`SO(3)` (see :func:`~pytransform3d.rotations.left_jacobian_SO3_inv`).

    Parameters
    ----------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    transform_log : array, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.
    """
    A2B = check_transform(A2B, strict_check=strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    transform_log = np.zeros((4, 4))

    if np.linalg.norm(np.eye(3) - R) < np.finfo(float).eps:
        transform_log[:3, 3] = p
        return transform_log

    omega_theta = compact_axis_angle_from_matrix(R)
    theta = np.linalg.norm(omega_theta)

    if theta == 0:
        return transform_log

    J_inv = left_jacobian_SO3_inv(omega_theta)
    v_theta = np.dot(J_inv, p)

    transform_log[:3, :3] = cross_product_matrix(omega_theta)
    transform_log[:3, 3] = v_theta

    return transform_log


def transform_from_exponential_coordinates(Stheta, check=True):
    r"""Compute transformation matrix from exponential coordinates.

    Exponential map.

    .. math::

        Exp: \mathcal{S} \theta \in \mathbb{R}^6
        \rightarrow \boldsymbol{T} \in SE(3)

    .. math::

        Exp(\mathcal{S}\theta) =
        Exp\left(\left(\begin{array}{c}
        \hat{\boldsymbol{\omega}}\\
        \boldsymbol{v}
        \end{array}\right)\theta\right)
        =
        \exp(\left[\mathcal{S}\right] \theta)
        =
        \left(\begin{array}{cc}
        Exp(\hat{\boldsymbol{\omega}} \theta) &
        \boldsymbol{J}(\theta)\boldsymbol{v}\theta\\
        \boldsymbol{0} & 1
        \end{array}\right),

    where :math:`\boldsymbol{J}(\theta)` is the left Jacobian of :math:`SO(3)`
    (see :func:`~pytransform3d.rotations.left_jacobian_SO3`).

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    check : bool, optional (default: True)
        Check if exponential coordinates are valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transformation matrix from frame A to frame B
    """
    if check:
        Stheta = check_exponential_coordinates(Stheta)

    omega_theta = Stheta[:3]
    theta = np.linalg.norm(omega_theta)

    if theta == 0.0:  # only translation
        return translate_transform(np.eye(4), Stheta[3:], check=check)

    omega_theta = Stheta[:3]
    v_theta = Stheta[3:]

    A2B = np.eye(4)
    A2B[:3, :3] = matrix_from_compact_axis_angle(omega_theta)
    J = left_jacobian_SO3(omega_theta)
    A2B[:3, 3] = np.dot(J, v_theta)
    return A2B


def transform_from_transform_log(transform_log):
    r"""Compute transformation from matrix logarithm of transformation.

    Exponential map.

    .. math::

        \exp: \left[ \mathcal{S} \right] \theta \in se(3)
        \rightarrow \boldsymbol{T} \in SE(3)

    .. math::

        \exp([\mathcal{S}]\theta) =
        \exp\left(\left(\begin{array}{cc}
        \left[\hat{\boldsymbol{\omega}}\right] & \boldsymbol{v}\\
        \boldsymbol{0} & 0
        \end{array}\right)\theta\right) =
        \left(\begin{array}{cc}
        Exp(\hat{\boldsymbol{\omega}} \theta) &
        \boldsymbol{J}(\theta)\boldsymbol{v}\theta\\
        \boldsymbol{0} & 1
        \end{array}\right),

    where :math:`\boldsymbol{J}(\theta)` is the left Jacobian of :math:`SO(3)`
    (see :func:`~pytransform3d.rotations.left_jacobian_SO3`).

    Parameters
    ----------
    transform_log : array-like, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    transform_log = check_transform_log(transform_log)

    omega_theta = np.array([
        transform_log[2, 1], transform_log[0, 2], transform_log[1, 0]])
    v_theta = transform_log[:3, 3]
    theta = np.linalg.norm(omega_theta)

    if theta == 0.0:  # only translation
        return translate_transform(np.eye(4), v_theta)

    A2B = np.eye(4)
    A2B[:3, :3] = matrix_from_compact_axis_angle(omega_theta)
    J = left_jacobian_SO3(omega_theta)
    A2B[:3, 3] = np.dot(J, v_theta)
    return A2B


def dual_quaternion_from_transform(A2B):
    """Compute dual quaternion from transformation matrix.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    A2B = check_transform(A2B)
    real = quaternion_from_matrix(A2B[:3, :3])
    dual = 0.5 * concatenate_quaternions(
        np.r_[0, A2B[:3, 3]], real)
    return np.hstack((real, dual))


def dual_quaternion_from_pq(pq):
    """Compute dual quaternion from position and quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    pq = check_pq(pq)
    real = pq[3:]
    dual = 0.5 * concatenate_quaternions(
        np.r_[0, pq[:3]], real)
    return np.hstack((real, dual))


def dual_quaternion_from_screw_parameters(q, s_axis, h, theta):
    """Compute dual quaternion from screw parameters.

    Parameters
    ----------
    q : array-like, shape (3,)
        Vector to a point on the screw axis

    s_axis : array-like, shape (3,)
        Direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    q, s_axis, h = check_screw_parameters(q, s_axis, h)

    if np.isinf(h):  # pure translation
        d = theta
        theta = 0
    else:
        d = h * theta
    moment = np.cross(q, s_axis)

    half_distance = 0.5 * d
    sin_half_angle = np.sin(0.5 * theta)
    cos_half_angle = np.cos(0.5 * theta)

    real_w = cos_half_angle
    real_vec = sin_half_angle * s_axis
    dual_w = -half_distance * sin_half_angle
    dual_vec = (sin_half_angle * moment +
                half_distance * cos_half_angle * s_axis)

    return np.r_[real_w, real_vec, dual_w, dual_vec]


def transform_from_dual_quaternion(dq):
    """Compute transformation matrix from dual quaternion.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    dq = check_dual_quaternion(dq)
    real = dq[:4]
    dual = dq[4:]
    R = matrix_from_quaternion(real)
    p = 2 * concatenate_quaternions(dual, q_conj(real))[1:]
    return transform_from(R=R, p=p)


def pq_from_dual_quaternion(dq):
    """Compute position and quaternion from dual quaternion.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    pq : array, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    dq = check_dual_quaternion(dq)
    real = dq[:4]
    dual = dq[4:]
    p = 2 * concatenate_quaternions(dual, q_conj(real))[1:]
    return np.hstack((p, real))


def screw_parameters_from_dual_quaternion(dq):
    """Compute screw parameters from dual quaternion.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    q : array, shape (3,)
        Vector to a point on the screw axis

    s_axis : array, shape (3,)
        Direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.
    """
    dq = check_dual_quaternion(dq, unit=True)

    real = dq[:4]
    dual = dq[4:]

    a = axis_angle_from_quaternion(real)
    s_axis = a[:3]
    theta = a[3]

    translation = 2 * concatenate_quaternions(dual, q_conj(real))[1:]
    if abs(theta) < np.finfo(float).eps:
        # pure translation
        d = np.linalg.norm(translation)
        if d < np.finfo(float).eps:
            s_axis = np.array([1, 0, 0])
        else:
            s_axis = translation / d
        q = np.zeros(3)
        theta = d
        h = np.inf
        return q, s_axis, h, theta

    distance = np.dot(translation, s_axis)
    moment = 0.5 * (np.cross(translation, s_axis) +
                    (translation - distance * s_axis)
                    / np.tan(0.5 * theta))
    dual = np.cross(s_axis, moment)
    h = distance / theta
    return dual, s_axis, h, theta


def adjoint_from_transform(A2B, strict_check=True, check=True):
    r"""Compute adjoint representation of a transformation matrix.

    The adjoint representation of a transformation
    :math:`\left[Ad_{\boldsymbol{T}_{BA}}\right] \in \mathbb{R}^{6 \times 6}`
    from frame A to frame B translates a twist from frame A to frame B
    through the adjoint map

    .. math::

        \mathcal{V}_{B}
        = \left[Ad_{\boldsymbol{T}_{BA}}\right] \mathcal{V}_A

    The corresponding transformation matrix operation is

    .. math::

        \left[\mathcal{V}_{B}\right]
        = \boldsymbol{T}_{BA} \left[\mathcal{V}_A\right]
        \boldsymbol{T}_{BA}^{-1}

    We can also use the adjoint representation to transform a wrench from frame
    A to frame B:

    .. math::

        \mathcal{F}_B
        = \left[ Ad_{\boldsymbol{T}_{AB}} \right]^T \mathcal{F}_A

    Note that not only the adjoint is transposed but also the transformation is
    inverted.

    Adjoint representations have the following properties:

    .. math::

        \left[Ad_{\boldsymbol{T}_1 \boldsymbol{T}_2}\right]
        = \left[Ad_{\boldsymbol{T}_1}\right]
        \left[Ad_{\boldsymbol{T}_2}\right]

    .. math::

        \left[Ad_{\boldsymbol{T}}\right]^{-1} =
        \left[Ad_{\boldsymbol{T}^{-1}}\right]

    For a transformation matrix

    .. math::

        \boldsymbol T =
        \left( \begin{array}{cc}
            \boldsymbol R & \boldsymbol t\\
            \boldsymbol 0 & 1\\
        \end{array} \right)

    the adjoint is defined as

    .. math::

        \left[Ad_{\boldsymbol{T}}\right]
        =
        \left( \begin{array}{cc}
            \boldsymbol R & \boldsymbol 0\\
            \left[\boldsymbol{t}\right]_{\times}\boldsymbol R & \boldsymbol R\\
        \end{array} \right),

    where :math:`\left[\boldsymbol{t}\right]_{\times}` is the cross-product
    matrix (see :func:`~pytransform3d.rotations.cross_product_matrix`) of the
    translation component.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    adj_A2B : array, shape (6, 6)
        Adjoint representation of transformation matrix
    """
    if check:
        A2B = check_transform(A2B, strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    adj_A2B = np.zeros((6, 6))
    adj_A2B[:3, :3] = R
    adj_A2B[3:, :3] = np.dot(cross_product_matrix(p), R)
    adj_A2B[3:, 3:] = R
    return adj_A2B


def norm_exponential_coordinates(Stheta):
    """Normalize exponential coordinates of transformation.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation. Theta should be >= 0. Negative rotations
        will be represented by a negative screw axis instead. This is relevant
        if you want to recover theta from exponential coordinates.

    Returns
    -------
    Stheta : array, shape (6,)
        Normalized exponential coordinates of transformation with theta in
        [0, pi]. Note that in the case of pure translation no normalization
        is required because the representation is unique. In the case of
        rotation by pi, there is an ambiguity that will be resolved so that
        the screw pitch is positive.
    """
    theta = np.linalg.norm(Stheta[:3])
    if theta == 0.0:
        return Stheta

    screw_axis = Stheta / theta
    q, s_axis, h = screw_parameters_from_screw_axis(screw_axis)
    if abs(theta - np.pi) < eps and h < 0:
        h *= -1.0
        s_axis *= -1.0
    theta_normed = norm_angle(theta)
    h_normalized = h * theta / theta_normed
    screw_axis = screw_axis_from_screw_parameters(q, s_axis, h_normalized)

    return screw_axis * theta_normed
