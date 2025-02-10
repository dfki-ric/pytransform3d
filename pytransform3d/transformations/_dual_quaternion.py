"""Dual quaternion operations."""
import numpy as np
from ._utils import check_screw_parameters
from ._transform import transform_from
from ..rotations import (
    concatenate_quaternions, q_conj, axis_angle_from_quaternion,
    matrix_from_quaternion)


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


def norm_dual_quaternion(dq):
    """Normalize unit dual quaternion.

    A unit dual quaternion has a real quaternion with unit norm and an
    orthogonal real part. Both properties are enforced by multiplying a
    normalization factor [1]_. This is not always necessary. It is often
    sufficient to only enforce the unit norm property of the real quaternion.
    This can also be done with :func:`check_dual_quaternion`.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform with orthogonal real and
        dual quaternion.

    See Also
    --------
    check_dual_quaternion
        Input validation of dual quaternion representation. Has an option to
        normalize the dual quaternion.

    dual_quaternion_requires_renormalization
        Check if normalization is required.

    References
    ----------
    .. [1] enki (2023). Properly normalizing a dual quaternion.
       https://stackoverflow.com/a/76313524
    """
    dq = check_dual_quaternion(dq, unit=False)
    dq_prod = concatenate_dual_quaternions(dq, dq_q_conj(dq), unit=False)

    prod_real = dq_prod[:4]
    prod_dual = dq_prod[4:]

    real = np.copy(dq[:4])
    dual = dq[4:]

    prod_real_norm = np.linalg.norm(prod_real)
    if prod_real_norm == 0.0:
        real = np.array([1.0, 0.0, 0.0, 0.0])
        prod_real_norm = 1.0
        valid_dq = np.hstack((real, dual))
        prod_dual = concatenate_dual_quaternions(
            valid_dq, dq_q_conj(valid_dq), unit=False)[4:]

    real_inv_sqrt = 1.0 / prod_real_norm
    dual_inv_sqrt = -0.5 * prod_dual * real_inv_sqrt ** 3

    real = real_inv_sqrt * real
    dual = real_inv_sqrt * dual + concatenate_quaternions(dual_inv_sqrt, real)

    return np.hstack((real, dual))


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


def dual_quaternion_double(dq):
    r"""Create another dual quaternion that represents the same transformation.

    The unit dual quaternions
    :math:`\boldsymbol{\sigma} = \boldsymbol{p} + \epsilon \boldsymbol{q}` and
    :math:`-\boldsymbol{\sigma}` represent exactly the same transformation.
    The reason for this ambiguity is that the real quaternion
    :math:`\boldsymbol{p}` represents the orientation component, the dual
    quaternion encodes the translation component as
    :math:`\boldsymbol{q} = 0.5 \boldsymbol{t} \boldsymbol{p}`, where
    :math:`\boldsymbol{t}` is a quaternion with the translation in the vector
    component and the scalar 0, and rotation quaternions have the same
    ambiguity.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_double : array, shape (8,)
        -dq
    """
    return -check_dual_quaternion(dq, unit=True)


def dq_conj(dq):
    """Conjugate of dual quaternion.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, -qw, qx, qy, qz). It is a combination of the quaternion
    conjugate and the dual number conjugate.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_conjugate : array, shape (8,)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, -qw, qx, qy, qz)

    See Also
    --------
    dq_q_conj
        Quaternion conjugate of dual quaternion.
    """
    dq = check_dual_quaternion(dq)
    return np.r_[dq[0], -dq[1:5], dq[5:]]


def dq_q_conj(dq):
    """Quaternion conjugate of dual quaternion.

    For unit dual quaternions that represent transformations, this function
    is equivalent to the inverse of the corresponding transformation matrix.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, qw, -qx, -qy, -qz). It is the quaternion conjugate
    applied to each of the two quaternions.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_q_conjugate : array, shape (8,)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, qw, -qx, -qy, -qz)

    See Also
    --------
    dq_conj
        Conjugate of a dual quaternion.
    """
    dq = check_dual_quaternion(dq)
    return np.r_[dq[0], -dq[1:4], dq[4], -dq[5:]]


def concatenate_dual_quaternions(dq1, dq2, unit=True):
    r"""Concatenate dual quaternions.

    We concatenate two dual quaternions by dual quaternion multiplication

    .. math::

        (\boldsymbol{p}_1 + \epsilon \boldsymbol{q}_1)
        (\boldsymbol{p}_2 + \epsilon \boldsymbol{q}_2)
        = \boldsymbol{p}_1 \boldsymbol{p}_2 + \epsilon (
        \boldsymbol{p}_1 \boldsymbol{q}_2 + \boldsymbol{q}_1 \boldsymbol{p}_2)

    using Hamilton multiplication of quaternions.

    .. warning::

        Note that the order of arguments is different than the order in
        :func:`concat`.

    Parameters
    ----------
    dq1 : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    dq2 : array-like, shape (8,)
        Dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    unit : bool, optional (default: True)
        Normalize the dual quaternion so that it is a unit dual quaternion.
        A unit dual quaternion has the properties
        :math:`p_w^2 + p_x^2 + p_y^2 + p_z^2 = 1` and
        :math:`p_w q_w + p_x q_x + p_y q_y + p_z q_z = 0`.

    Returns
    -------
    dq3 : array, shape (8,)
        Product of the two dual quaternions:
        (pw, px, py, pz, qw, qx, qy, qz)

    See Also
    --------
    pytransform3d.rotations.concatenate_quaternions
        Quaternion multiplication.
    """
    dq1 = check_dual_quaternion(dq1, unit=unit)
    dq2 = check_dual_quaternion(dq2, unit=unit)
    real = concatenate_quaternions(dq1[:4], dq2[:4])
    dual = (concatenate_quaternions(dq1[:4], dq2[4:]) +
            concatenate_quaternions(dq1[4:], dq2[:4]))
    return np.hstack((real, dual))


def dq_prod_vector(dq, v):
    r"""Apply transform represented by a dual quaternion to a vector.

    To apply the transformation defined by a unit dual quaternion
    :math:`\boldsymbol{q}` to a point :math:`\boldsymbol{v} \in \mathbb{R}^3`,
    we first represent the vector as a dual quaternion: we set the real part to
    (1, 0, 0, 0) and the dual part is a pure quaternion with the scalar part
    0 and the vector as its vector part
    :math:`\left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right) \in
    \mathbb{R}^4`. Then we left-multiply the dual quaternion and right-multiply
    its dual quaternion conjugate

    .. math::

        \left(\begin{array}{c}1\\0\\0\\0\\0\\\boldsymbol{w}\end{array}\right)
        =
        \boldsymbol{q}
        \cdot
        \left(\begin{array}{c}1\\0\\0\\0\\0\\\boldsymbol{v}\end{array}\right)
        \cdot
        \boldsymbol{q}^*.

    The vector part of the dual part :math:`\boldsymbol{w}` of the resulting
    quaternion is the rotated point.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    w : array, shape (3,)
        3d vector
    """
    dq = check_dual_quaternion(dq)
    v_dq = np.r_[1, 0, 0, 0, 0, v]
    v_dq_transformed = concatenate_dual_quaternions(
        concatenate_dual_quaternions(dq, v_dq),
        dq_conj(dq))
    return v_dq_transformed[5:]


def dual_quaternion_sclerp(start, end, t):
    """Screw linear interpolation (ScLERP) for dual quaternions.

    Although linear interpolation of dual quaternions is possible, this does
    not result in constant velocities. If you want to generate interpolations
    with constant velocity, you have to use ScLERP.

    Parameters
    ----------
    start : array-like, shape (8,)
        Unit dual quaternion to represent start pose:
        (pw, px, py, pz, qw, qx, qy, qz)

    end : array-like, shape (8,)
        Unit dual quaternion to represent end pose:
        (pw, px, py, pz, qw, qx, qy, qz)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    dq_t : array, shape (8,)
        Interpolated unit dual quaternion: (pw, px, py, pz, qw, qx, qy, qz)

    References
    ----------
    .. [1] Kavan, L., Collins, S., O'Sullivan, C., Zara, J. (2006).
       Dual Quaternions for Rigid Transformation Blending, Technical report,
       Trinity College Dublin,
       https://users.cs.utah.edu/~ladislav/kavan06dual/kavan06dual.pdf

    See Also
    --------
    transform_sclerp :
        ScLERP for transformation matrices.

    pq_slerp :
        An alternative approach is spherical linear interpolation (SLERP) with
        position and quaternion.
    """
    start = check_dual_quaternion(start)
    end = check_dual_quaternion(end)
    diff = concatenate_dual_quaternions(dq_q_conj(start), end)
    return concatenate_dual_quaternions(start, dual_quaternion_power(diff, t))


def dual_quaternion_power(dq, t):
    r"""Compute power of unit dual quaternion with respect to scalar.

    .. math::

        (p + \epsilon q)^t

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    t : float
        Exponent

    Returns
    -------
    dq_t : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz) ** t
    """
    dq = check_dual_quaternion(dq)
    q, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    return dual_quaternion_from_screw_parameters(q, s_axis, h, theta * t)


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
