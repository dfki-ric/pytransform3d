"""Quaternion operations."""
import numpy as np
from ._utils import check_quaternion, check_quaternions
from ._conversions import (quaternion_from_compact_axis_angle,
                           compact_axis_angle_from_quaternion,
                           axis_angle_from_quaternion)


def quaternion_integrate(Qd, q0=np.array([1.0, 0.0, 0.0, 0.0]), dt=1.0):
    """Integrate angular velocities to quaternions.

    Parameters
    ----------
    Qd : array-like, shape (n_steps, 3)
        Angular velocities in a compact axis-angle representation. Each angular
        velocity represents the rotational offset after one unit of time.

    q0 : array-like, shape (4,), optional (default: [1, 0, 0, 0])
        Unit quaternion to represent initial rotation: (w, x, y, z)

    dt : float, optional (default: 1)
        Time interval between steps.

    Returns
    -------
    Q : array-like, shape (n_steps, 4)
        Quaternions to represent rotations: (w, x, y, z)
    """
    Q = np.empty((len(Qd), 4))
    Q[0] = q0
    for t in range(1, len(Qd)):
        qd = (Qd[t] + Qd[t - 1]) / 2.0
        Q[t] = concatenate_quaternions(
            quaternion_from_compact_axis_angle(dt * qd), Q[t - 1])
    return Q


def quaternion_gradient(Q, dt=1.0):
    """Time-derivatives of a sequence of quaternions.

    Note that this function does not provide the exact same functionality for
    quaternions as `NumPy's gradient function
    <https://numpy.org/doc/stable/reference/generated/numpy.gradient.html>`_
    for positions. Gradients are always computed as central differences except
    the first and last gradient. We additionally accept a parameter dt that
    defines the time interval between each quaternion. Note that this means
    that we expect this to be constant for the whole sequence.

    Parameters
    ----------
    Q : array-like, shape (n_steps, 4)
        Quaternions to represent rotations: (w, x, y, z)

    dt : float, optional (default: 1)
        Time interval between steps. If you have non-constant dt, you can pass
        1 and manually divide angular velocities by their corresponding time
        interval afterwards.

    Returns
    -------
    A : array-like, shape (n_steps, 3)
        Angular velocities in a compact axis-angle representation. Each angular
        velocity represents the rotational offset after one unit of time.
    """
    Q = check_quaternions(Q)
    Qd = np.empty((len(Q), 3))
    Qd[0] = compact_axis_angle_from_quaternion(
        concatenate_quaternions(Q[1], q_conj(Q[0]))) / dt
    for t in range(1, len(Q) - 1):
        # divided by two because of central differences
        Qd[t] = compact_axis_angle_from_quaternion(
            concatenate_quaternions(Q[t + 1], q_conj(Q[t - 1]))) / (2.0 * dt)
    Qd[-1] = compact_axis_angle_from_quaternion(
        concatenate_quaternions(Q[-1], q_conj(Q[-2]))) / dt
    return Qd


def concatenate_quaternions(q1, q2):
    """Concatenate two quaternions.

    We use Hamilton's quaternion multiplication.

    Suppose we want to apply two extrinsic rotations given by quaternions
    q1 and q2 to a vector v. We can either apply q2 to v and then q1 to
    the result or we can concatenate q1 and q2 and apply the result to v.

    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-like, shape (4,)
        Second quaternion

    Returns
    -------
    q12 : array-like, shape (4,)
        Quaternion that represents the concatenated rotation q1 * q2
    """
    q1 = check_quaternion(q1, unit=False)
    q2 = check_quaternion(q2, unit=False)
    q12 = np.empty(4)
    q12[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    q12[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    return q12


def q_prod_vector(q, v):
    """Apply rotation represented by a quaternion to a vector.

    We use Hamilton's quaternion multiplication.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    w : array-like, shape (3,)
        3d vector
    """
    q = check_quaternion(q)
    t = 2 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def q_conj(q):
    """Conjugate of quaternion.

    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion. The conjugate of a quaternion q is often denoted
    as q*.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    q_c : array-like, shape (4,)
        Conjugate (w, -x, -y, -z)
    """
    q = check_quaternion(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_dist(q1, q2):
    r"""Compute distance between two quaternions.

    We use the angular metric of :math:`S^3`, which is defined as

    .. math::

        d(q_1, q_2) = \min(|| \log(q_1 * \overline{q_2})||,
                            2 \pi - || \log(q_1 * \overline{q_2})||)

    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-like, shape (4,)
        Second quaternion

    Returns
    -------
    dist : float
        Distance between q1 and q2
    """
    q1 = check_quaternion(q1)
    q2 = check_quaternion(q2)
    q12c = concatenate_quaternions(q1, q_conj(q2))
    angle = axis_angle_from_quaternion(q12c)[-1]
    return min(angle, 2.0 * np.pi - angle)


def quaternion_diff(q1, q2):
    r"""Compute the rotation in angle-axis format that rotates q2 into q1.

    .. math::

        \omega = 2 \log (q_1 * \overline{q_2})

    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-line, shape (4,)
        Second quaternion

    Returns
    -------
    a : array-like, shape (4,)
        The rotation in angle-axis format that rotates q2 into q1
    """
    q1 = check_quaternion(q1)
    q2 = check_quaternion(q2)
    q1q2c = concatenate_quaternions(q1, q_conj(q2))
    return axis_angle_from_quaternion(q1q2c)
