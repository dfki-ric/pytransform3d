"""Dual quaternion operations."""
import numpy as np
from ._utils import check_dual_quaternion
from ._conversions import (screw_parameters_from_dual_quaternion,
                           dual_quaternion_from_screw_parameters)
from ..rotations import concatenate_quaternions


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


def concatenate_dual_quaternions(dq1, dq2):
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
    dq1 = check_dual_quaternion(dq1)
    dq2 = check_dual_quaternion(dq2)
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
