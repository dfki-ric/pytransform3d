"""Conversions between transform representations."""
import numpy as np
from ._utils import (
    check_screw_axis, check_screw_parameters, check_exponential_coordinates,
    check_screw_matrix, check_transform_log)
from ._transform import translate_transform
from ..rotations import (
    matrix_from_compact_axis_angle, cross_product_matrix, left_jacobian_SO3)


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
