"""Conversions between transform representations."""
import numpy as np
from numpy.testing import assert_array_almost_equal
from ._transform import translate_transform
from ..rotations import (
    eps, norm_vector, norm_angle, cross_product_matrix,
    matrix_from_compact_axis_angle, check_skew_symmetric_matrix,
    left_jacobian_SO3)


def check_screw_parameters(q, s_axis, h):
    r"""Input validation of screw parameters.

    The parameters :math:`(\boldsymbol{q}, \hat{\boldsymbol{s}}, h)`
    describe a screw.

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
    q : array, shape (3,)
        Vector to a point on the screw axis. Will be set to zero vector when
        pitch is infinite (pure translation).

    s_axis : array, shape (3,)
        Unit direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    Raises
    ------
    ValueError
        If input is invalid
    """
    s_axis = np.asarray(s_axis, dtype=np.float64)
    if s_axis.ndim != 1 or s_axis.shape[0] != 3:
        raise ValueError("Expected 3D vector with shape (3,), got array-like "
                         "object with shape %s" % (s_axis.shape,))
    if np.linalg.norm(s_axis) == 0.0:
        raise ValueError("s_axis must not have norm 0")

    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] != 3:
        raise ValueError("Expected 3D vector with shape (3,), got array-like "
                         "object with shape %s" % (q.shape,))
    if np.isinf(h):  # pure translation
        q = np.zeros(3)

    return q, norm_vector(s_axis), h


def check_screw_axis(screw_axis):
    r"""Input validation of screw axis.

    A screw axis

    .. math::

        \mathcal{S}
        = \left[\begin{array}{c}\boldsymbol{\omega}\\
          \boldsymbol{v}\end{array}\right] \in \mathbb{R}^6

    consists of a part that describes rotation and a part that describes
    translation.

    Parameters
    ----------
    screw_axis : array-like, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    Returns
    -------
    screw_axis : array, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    Raises
    ------
    ValueError
        If input is invalid
    """
    screw_axis = np.asarray(screw_axis, dtype=np.float64)
    if screw_axis.ndim != 1 or screw_axis.shape[0] != 6:
        raise ValueError("Expected 3D vector with shape (6,), got array-like "
                         "object with shape %s" % (screw_axis.shape,))

    omega_norm = np.linalg.norm(screw_axis[:3])
    if (abs(omega_norm - 1.0) > 10.0 * np.finfo(float).eps
            and abs(omega_norm) > 10.0 * np.finfo(float).eps):
        raise ValueError(
            "Norm of rotation axis must either be 0 or 1, but it is %g."
            % omega_norm)
    if abs(omega_norm) < np.finfo(float).eps:
        v_norm = np.linalg.norm(screw_axis[3:])
        if abs(v_norm - 1.0) > np.finfo(float).eps:
            raise ValueError(
                "If the norm of the rotation axis is 0, then the direction "
                "vector must have norm 1, but it is %g." % v_norm)

    return screw_axis


def check_exponential_coordinates(Stheta):
    """Input validation for exponential coordinates of transformation.

    Exponential coordinates of a transformation :math:`\\mathcal{S}\\theta
    \\in \\mathbb{R}^6` are the product of a screw axis and a scalar
    :math:`\\theta`.

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
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation. Theta should be >= 0. Negative rotations
        will be represented by a negative screw axis instead. This is relevant
        if you want to recover theta from exponential coordinates.

    Raises
    ------
    ValueError
        If input is invalid
    """
    Stheta = np.asarray(Stheta, dtype=np.float64)
    if Stheta.ndim != 1 or Stheta.shape[0] != 6:
        raise ValueError("Expected array-like with shape (6,), got array-like "
                         "object with shape %s" % (Stheta.shape,))
    return Stheta


def check_screw_matrix(screw_matrix, tolerance=1e-6, strict_check=True):
    """Input validation for screw matrix.

    A screw matrix consists of the cross-product matrix of a rotation
    axis and a translation.

    .. math::

        \\left[\\mathcal S\\right]
        =
        \\left( \\begin{array}{cc}
            \\left[\\boldsymbol{\\omega}\\right] & \\boldsymbol v\\\\
            \\boldsymbol 0 & 0\\\\
        \\end{array} \\right)
        =
        \\left(
        \\begin{matrix}
        0 & -\\omega_3 & \\omega_2 & v_1\\\\
        \\omega_3 & 0 & -\\omega_1 & v_2\\\\
        -\\omega_2 & \\omega_1 & 0 & v_3\\\\
        0 & 0 & 0 & 0\\\\
        \\end{matrix}
        \\right)
        \\in se(3) \\subset \\mathbb{R}^{4 \\times 4}

    Parameters
    ----------
    screw_matrix : array-like, shape (4, 4)
        A screw matrix consists of a cross-product matrix that represents an
        axis of rotation, a translation, and a row of zeros.

    tolerance : float, optional (default: 1e-6)
        Tolerance threshold for checks.

    strict_check : bool, optional (default: True)
        Raise a ValueError if [omega].T is not numerically close enough to
        -[omega]. Otherwise we print a warning.

    Returns
    -------
    screw_matrix : array, shape (4, 4)
        A screw matrix consists of a cross-product matrix that represents an
        axis of rotation, a translation, and a row of zeros.

    Raises
    ------
    ValueError
        If input is invalid
    """
    screw_matrix = np.asarray(screw_matrix, dtype=np.float64)
    if (screw_matrix.ndim != 2 or screw_matrix.shape[0] != 4
            or screw_matrix.shape[1] != 4):
        raise ValueError(
            "Expected array-like with shape (4, 4), got array-like "
            "object with shape %s" % (screw_matrix.shape,))
    if any(screw_matrix[3] != 0.0):
        raise ValueError("Last row of screw matrix must only contains zeros.")

    check_skew_symmetric_matrix(screw_matrix[:3, :3], tolerance, strict_check)

    omega_norm = np.linalg.norm(
        [screw_matrix[2, 1], screw_matrix[0, 2], screw_matrix[1, 0]])

    if (abs(omega_norm - 1.0) > np.finfo(float).eps
            and abs(omega_norm) > np.finfo(float).eps):
        raise ValueError(
            "Norm of rotation axis must either be 0 or 1, but it is %g."
            % omega_norm)
    if abs(omega_norm) < np.finfo(float).eps:
        v_norm = np.linalg.norm(screw_matrix[:3, 3])
        if (abs(v_norm - 1.0) > np.finfo(float).eps
                and abs(v_norm) > np.finfo(float).eps):
            raise ValueError(
                "If the norm of the rotation axis is 0, then the direction "
                "vector must have norm 1 or 0, but it is %g." % v_norm)

    return screw_matrix


def check_transform_log(transform_log, tolerance=1e-6, strict_check=True):
    """Input validation for logarithm of transformation.

    The logarithm of a transformation :math:`\\left[\\mathcal{S}\\right]\\theta
    \\in se(3) \\subset \\mathbb{R}^{4 \\times 4}` are the product of a screw
    matrix and a scalar :math:`\\theta`.

    Parameters
    ----------
    transform_log : array-like, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.

    tolerance : float, optional (default: 1e-6)
        Tolerance threshold for checks.

    strict_check : bool, optional (default: True)
        Raise a ValueError if [omega].T is not numerically close enough to
        -[omega]. Otherwise we print a warning.

    Returns
    -------
    transform_log : array, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.

    Raises
    ------
    ValueError
        If input is invalid
    """
    transform_log = np.asarray(transform_log, dtype=np.float64)
    if (transform_log.ndim != 2 or transform_log.shape[0] != 4
            or transform_log.shape[1] != 4):
        raise ValueError(
            "Expected array-like with shape (4, 4), got array-like "
            "object with shape %s" % (transform_log.shape,))
    if any(transform_log[3] != 0.0):
        raise ValueError(
            "Last row of logarithm of transformation must only "
            "contains zeros.")

    check_skew_symmetric_matrix(transform_log[:3, :3], tolerance, strict_check)

    return transform_log



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


def assert_exponential_coordinates_equal(Stheta1, Stheta2):
    """Raise an assertion if exp. coordinates are not approximately equal.

    Parameters
    ----------
    Stheta1 : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    Stheta2 : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    Stheta1 = norm_exponential_coordinates(Stheta1)
    Stheta2 = norm_exponential_coordinates(Stheta2)
    assert_array_almost_equal(Stheta1, Stheta2)


def assert_screw_parameters_equal(
        q1, s_axis1, h1, theta1, q2, s_axis2, h2, theta2, *args, **kwargs):
    """Raise an assertion if two sets of screw parameters are not similar.

    Note that the screw axis can be inverted. In this case theta and h have
    to be adapted.

    Parameters
    ----------
    q1 : array, shape (3,)
        Vector to a point on the screw axis that is orthogonal to s_axis

    s_axis1 : array, shape (3,)
        Unit direction vector of the screw axis

    h1 : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta1 : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    q2 : array, shape (3,)
        Vector to a point on the screw axis that is orthogonal to s_axis

    s_axis2 : array, shape (3,)
        Unit direction vector of the screw axis

    h2 : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta2 : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    # normalize thetas
    theta1_new = norm_angle(theta1)
    h1 *= theta1 / theta1_new
    theta1 = theta1_new

    theta2_new = norm_angle(theta2)
    h2 *= theta2 / theta2_new
    theta2 = theta2_new

    # q1 and q2 can be any points on the screw axis, that is, they must be a
    # linear combination of each other and the screw axis (which one does not
    # matter since they should be identical or mirrored)
    q1_to_q2 = q2 - q1
    factors = q1_to_q2 / s_axis2
    assert abs(factors[0] - factors[1]) < eps
    assert abs(factors[1] - factors[2]) < eps
    try:
        assert_array_almost_equal(s_axis1, s_axis2, *args, **kwargs)
        assert abs(h1 - h2) < eps
        assert abs(theta1 - theta2) < eps
    except AssertionError:  # possibly mirrored screw axis
        s_axis1_new = -s_axis1
        # make sure that we keep the direction of rotation
        theta1_new = 2.0 * np.pi - theta1
        # adjust pitch: switch sign and update rotation component
        h1 = -h1 / theta1_new * theta1
        theta1 = theta1_new

        # we have to normalize the angle again
        theta1_new = norm_angle(theta1)
        h1 *= theta1 / theta1_new
        theta1 = theta1_new

        assert_array_almost_equal(s_axis1_new, s_axis2, *args, **kwargs)
        assert abs(h1 - h2) < eps
        assert abs(theta1 - theta2) < eps


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
