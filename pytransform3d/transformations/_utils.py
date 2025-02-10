"""Utility functions for transforms."""
import numpy as np
from ..rotations import norm_vector, check_skew_symmetric_matrix


def check_pq(pq):
    """Input validation for position and orientation quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    pq : array, shape (7,)
        Validated position and orientation quaternion:
         (x, y, z, qw, qx, qy, qz)

    Raises
    ------
    ValueError
        If input is invalid
    """
    pq = np.asarray(pq, dtype=np.float64)
    if pq.ndim != 1 or pq.shape[0] != 7:
        raise ValueError("Expected position and orientation quaternion in a "
                         "1D array, got array-like object with shape %s"
                         % (pq.shape,))
    return pq


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


def dual_quaternion_requires_renormalization(dq, tolerance=1e-6):
    r"""Check if dual quaternion requires renormalization.

    Dual quaternions that represent transformations in 3D should have unit
    norm. Since the real and the dual quaternion are orthogonal, their dot
    product should be 0. In addition, :math:`\epsilon^2 = 0`. Hence,

    .. math::

        ||\boldsymbol{p} + \epsilon \boldsymbol{q}||
        =
        \sqrt{(\boldsymbol{p} + \epsilon \boldsymbol{q}) \cdot
        (\boldsymbol{p} + \epsilon \boldsymbol{q})}
        =
        \sqrt{\boldsymbol{p}\cdot\boldsymbol{p}
        + 2\epsilon \boldsymbol{p}\cdot\boldsymbol{q}
        + \epsilon^2 \boldsymbol{q}\cdot\boldsymbol{q}}
        =
        \sqrt{\boldsymbol{p}\cdot\boldsymbol{p}},

    i.e., the norm only depends on the real quaternion.

    This function checks unit norm and orthogonality of the real and dual
    part.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    tolerance : float, optional (default: 1e-6)
        Tolerance for check.

    Returns
    -------
    required : bool
        Indicates if renormalization is required.

    See Also
    --------
    check_dual_quaternion
        Input validation of dual quaternion representation. Has an option to
        normalize the dual quaternion.

    norm_dual_quaternion
        Normalization that enforces unit norm and orthogonality of the real and
        dual quaternion.

    assert_unit_dual_quaternion
        Checks unit norm and orthogonality of real and dual quaternion.
    """
    real = dq[:4]
    dual = dq[4:]
    real_norm = np.linalg.norm(real)
    real_dual_dot = np.dot(real, dual)
    return abs(real_norm - 1.0) > tolerance or abs(real_dual_dot) > tolerance


def check_dual_quaternion(dq, unit=True):
    """Input validation of dual quaternion representation.

    See http://web.cs.iastate.edu/~cs577/handouts/dual-quaternion.pdf

    A dual quaternion is defined as

    .. math::

        \\boldsymbol{\\sigma} = \\boldsymbol{p} + \\epsilon \\boldsymbol{q},

    where :math:`\\boldsymbol{p}` and :math:`\\boldsymbol{q}` are both
    quaternions and :math:`\\epsilon` is the dual unit with
    :math:`\\epsilon^2 = 0`. The first quaternion is also called the real part
    and the second quaternion is called the dual part.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    unit : bool, optional (default: True)
        Normalize the dual quaternion so that it is a unit dual quaternion.
        A unit dual quaternion has the properties
        :math:`p_w^2 + p_x^2 + p_y^2 + p_z^2 = 1` and
        :math:`p_w q_w + p_x q_x + p_y q_y + p_z q_z = 0`.

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Raises
    ------
    ValueError
        If input is invalid

    See Also
    --------
    norm_dual_quaternion
        Normalization that enforces unit norm and orthogonality of the real and
        dual quaternion.
    """
    dq = np.asarray(dq, dtype=np.float64)
    if dq.ndim != 1 or dq.shape[0] != 8:
        raise ValueError("Expected dual quaternion with shape (8,), got "
                         "array-like object with shape %s" % (dq.shape,))
    if unit:
        # Norm of a dual quaternion only depends on the real part because
        # the dual part vanishes with (1) epsilon ** 2 = 0 and (2) the real
        # and dual part being orthogonal, i.e., their product is 0.
        real_norm = np.linalg.norm(dq[:4])
        if real_norm == 0.0:
            return np.r_[1, 0, 0, 0, dq[4:]]
        return dq / real_norm
    return dq
