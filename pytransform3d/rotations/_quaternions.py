"""Quaternion operations."""
import numpy as np
from ._utils import check_axis_index, norm_vector
from ._angle import quaternion_from_angle
from ._axis_angle import (
    norm_axis_angle, quaternion_from_compact_axis_angle, compact_axis_angle)


def quaternion_requires_renormalization(q, tolerance=1e-6):
    r"""Check if a unit quaternion requires renormalization.

    Quaternions that represent rotations should have unit norm, so we check
    :math:`||\boldsymbol{q}|| \approx 1`.

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion to represent rotation: (w, x, y, z)

    tolerance : float, optional (default: 1e-6)
        Tolerance for check.

    Returns
    -------
    required : bool
        Renormalization is required.

    See Also
    --------
    check_quaternion : Normalizes quaternion.
    """
    return abs(np.linalg.norm(q) - 1.0) > tolerance


def check_quaternion(q, unit=True):
    """Input validation of quaternion representation.

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion to represent rotation: (w, x, y, z)

    unit : bool, optional (default: True)
        Normalize the quaternion so that it is a unit quaternion

    Returns
    -------
    q : array, shape (4,)
        Validated quaternion to represent rotation: (w, x, y, z)

    Raises
    ------
    ValueError
        If input is invalid
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.shape[0] != 4:
        raise ValueError("Expected quaternion with shape (4,), got "
                         "array-like object with shape %s" % (q.shape,))
    if unit:
        return norm_vector(q)
    return q


def check_quaternions(Q, unit=True):
    """Input validation of quaternion representation.

    Parameters
    ----------
    Q : array-like, shape (n_steps, 4)
        Quaternions to represent rotations: (w, x, y, z)

    unit : bool, optional (default: True)
        Normalize the quaternions so that they are unit quaternions

    Returns
    -------
    Q : array, shape (n_steps, 4)
        Validated quaternions to represent rotations: (w, x, y, z)

    Raises
    ------
    ValueError
        If input is invalid
    """
    Q_checked = np.asarray(Q, dtype=np.float64)
    if Q_checked.ndim != 2 or Q_checked.shape[1] != 4:
        raise ValueError(
            "Expected quaternion array with shape (n_steps, 4), got "
            "array-like object with shape %s" % (Q_checked.shape,))
    if unit:
        for i in range(len(Q)):
            Q_checked[i] = norm_vector(Q_checked[i])
    return Q_checked


def quaternion_double(q):
    """Create another quaternion that represents the same orientation.

    The unit quaternions q and -q represent the same orientation (double
    cover).

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion.

    Returns
    -------
    q_double : array, shape (4,)
        -q

    See Also
    --------
    pick_closest_quaternion
        Picks the quaternion that is closest to another one in Euclidean space.
    """
    return -check_quaternion(q, unit=True)


def quaternion_integrate(Qd, q0=np.array([1.0, 0.0, 0.0, 0.0]), dt=1.0):
    """Integrate angular velocities to quaternions.

     Angular velocities are given in global frame and will be left-multiplied
     to the initial or previous orientation respectively.

    Parameters
    ----------
    Qd : array-like, shape (n_steps, 3)
        Angular velocities in a compact axis-angle representation. Each angular
        velocity represents the rotational offset after one unit of time.

    q0 : array-like, shape (4,), optional (default: [1, 0, 0, 0])
        Unit quaternion to represent initial orientation: (w, x, y, z)

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
    A : array, shape (n_steps, 3)
        Angular velocities in a compact axis-angle representation. Each angular
        velocity represents the rotational offset after one unit of time.
        Angular velocities are given in global frame and will be
        left-multiplied during integration to the initial or previous
        orientation respectively.
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
    r"""Concatenate two quaternions.

    We concatenate two quaternions by quaternion multiplication
    :math:`\boldsymbol{q}_1\boldsymbol{q}_2`.

    We use Hamilton's quaternion multiplication.

    If the two quaternions are divided up into scalar part and vector part
    each, i.e.,
    :math:`\boldsymbol{q} = (w, \boldsymbol{v}), w \in \mathbb{R},
    \boldsymbol{v} \in \mathbb{R}^3`, then the quaternion product is

    .. math::

        \boldsymbol{q}_{12} =
        (w_1 w_2 - \boldsymbol{v}_1 \cdot \boldsymbol{v}_2,
        w_1 \boldsymbol{v}_2 + w_2 \boldsymbol{v}_1
        + \boldsymbol{v}_1 \times \boldsymbol{v}_2)

    with the scalar product :math:`\cdot` and the cross product :math:`\times`.

    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-like, shape (4,)
        Second quaternion

    Returns
    -------
    q12 : array, shape (4,)
        Quaternion that represents the concatenated rotation q1 * q2

    See Also
    --------
    concatenate_rotors : Concatenate rotors, which is the same operation.
    """
    q1 = check_quaternion(q1, unit=False)
    q2 = check_quaternion(q2, unit=False)
    q12 = np.empty(4)
    q12[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    q12[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    return q12


def q_prod_vector(q, v):
    r"""Apply rotation represented by a quaternion to a vector.

    We use Hamilton's quaternion multiplication.

    To apply the rotation defined by a unit quaternion :math:`\boldsymbol{q}
    \in S^3` to a vector :math:`\boldsymbol{v} \in \mathbb{R}^3`, we
    first represent the vector as a quaternion: we set the scalar part to 0 and
    the vector part is exactly the original vector
    :math:`\left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right) \in
    \mathbb{R}^4`. Then we left-multiply the quaternion and right-multiply
    its conjugate

    .. math::

        \left(\begin{array}{c}0\\\boldsymbol{w}\end{array}\right)
        =
        \boldsymbol{q}
        \cdot
        \left(\begin{array}{c}0\\\boldsymbol{v}\end{array}\right)
        \cdot
        \boldsymbol{q}^*.

    The vector part :math:`\boldsymbol{w}` of the resulting quaternion is
    the rotated vector.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    w : array, shape (3,)
        3d vector

    See Also
    --------
    rotor_apply
        The same operation with a different name.

    concatenate_quaternions
        Hamilton's quaternion multiplication.
    """
    q = check_quaternion(q)
    t = 2 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def q_conj(q):
    r"""Conjugate of quaternion.

    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion.

    The conjugate of a quaternion :math:`\boldsymbol{q}` is often denoted as
    :math:`\boldsymbol{q}^*`. For a quaternion :math:`\boldsymbol{q} = w
    + x \boldsymbol{i} + y \boldsymbol{j} + z \boldsymbol{k}` it is defined as

    .. math::

        \boldsymbol{q}^* = w - x \boldsymbol{i} - y \boldsymbol{j}
        - z \boldsymbol{k}.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    q_c : array-like, shape (4,)
        Conjugate (w, -x, -y, -z)

    See Also
    --------
    rotor_reverse : Reverse of a rotor, which is the same operation.
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

        (\hat{\boldsymbol{\omega}}, \theta) = \log (q_1 \overline{q_2})

    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-line, shape (4,)
        Second quaternion

    Returns
    -------
    a : array, shape (4,)
        The rotation in angle-axis format that rotates q2 into q1
    """
    q1 = check_quaternion(q1)
    q2 = check_quaternion(q2)
    q1q2c = concatenate_quaternions(q1, q_conj(q2))
    return axis_angle_from_quaternion(q1q2c)


def quaternion_from_euler(e, i, j, k, extrinsic):
    """General conversion to quaternion from any Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order. The
        first and last angle are normalized to [-pi, pi]. The middle angle is
        normalized to either [0, pi] (proper Euler angles) or [-pi/2, pi/2]
        (Cardan / Tait-Bryan angles).

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
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Raises
    ------
    ValueError
        If basis is invalid
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    q0 = quaternion_from_angle(i, e[0])
    q1 = quaternion_from_angle(j, e[1])
    q2 = quaternion_from_angle(k, e[2])
    if not extrinsic:
        q0, q2 = q2, q0
    return concatenate_quaternions(concatenate_quaternions(q2, q1), q0)


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
    q = check_quaternion(q, unit=True)
    w, x, y, z = q
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
