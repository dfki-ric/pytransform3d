"""Dual quaternion operations."""
import numpy as np
from ._utils import check_dual_quaternion
from ._conversions import (screw_parameters_from_dual_quaternion,
                           dual_quaternion_from_screw_parameters)
from ..rotations import concatenate_quaternions


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
    dq_conjugate : array-like, shape (8,)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, -qw, qx, qy, qz)
    """
    dq = check_dual_quaternion(dq)
    return np.r_[dq[0], -dq[1:5], dq[5:]]


def dq_q_conj(dq):
    """Quaternion conjugate of dual quaternion.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, qw, -qx, -qy, -qz). It is the quaternion conjugate
    applied to each of the two quaternions.

    For unit dual quaternions that represent transformations, this function
    is equivalent to the inverse of the corresponding transformation matrix.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_q_conjugate : array-like, shape (8,)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, qw, -qx, -qy, -qz)
    """
    dq = check_dual_quaternion(dq)
    return np.r_[dq[0], -dq[1:4], dq[4], -dq[5:]]


def concatenate_dual_quaternions(dq1, dq2):
    """Concatenate dual quaternions.

    Suppose we want to apply two extrinsic transforms given by dual
    quaternions dq1 and dq2 to a vector v. We can either apply dq2 to v and
    then dq1 to the result or we can concatenate dq1 and dq2 and apply the
    result to v.

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
    """
    dq1 = check_dual_quaternion(dq1)
    dq2 = check_dual_quaternion(dq2)
    real = concatenate_quaternions(dq1[:4], dq2[:4])
    dual = (concatenate_quaternions(dq1[:4], dq2[4:]) +
            concatenate_quaternions(dq1[4:], dq2[:4]))
    return np.hstack((real, dual))


def dq_prod_vector(dq, v):
    """Apply transform represented by a dual quaternion to a vector.

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
    a : array, shape (8,)
        Interpolated unit dual quaternion: (pw, px, py, pz, qw, qx, qy, qz)
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
    dq_t : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz) ** t
    """
    dq = check_dual_quaternion(dq)
    q, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    return dual_quaternion_from_screw_parameters(q, s_axis, h, theta * t)
