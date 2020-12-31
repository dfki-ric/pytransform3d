"""Rotations in three dimensions - SO(3).

See :doc:`rotations` for more information.
"""
import warnings
import math
import numpy as np
from numpy.testing import assert_array_almost_equal


unitx = np.array([1.0, 0.0, 0.0])
unity = np.array([0.0, 1.0, 0.0])
unitz = np.array([0.0, 0.0, 1.0])

R_id = np.eye(3)
a_id = np.array([1.0, 0.0, 0.0, 0.0])
q_id = np.array([1.0, 0.0, 0.0, 0.0])
q_i = np.array([0.0, 1.0, 0.0, 0.0])
q_j = np.array([0.0, 0.0, 1.0, 0.0])
q_k = np.array([0.0, 0.0, 0.0, 1.0])
e_xyz_id = np.array([0.0, 0.0, 0.0])
e_zyx_id = np.array([0.0, 0.0, 0.0])
p0 = np.array([0.0, 0.0, 0.0])

eps = 1e-7


def norm_vector(v):
    """Normalize vector.

    Parameters
    ----------
    v : array-like, shape (n,)
        nd vector

    Returns
    -------
    u : array, shape (n,)
        nd unit vector with norm 1 or the zero vector
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    else:
        return np.asarray(v) / norm


def norm_matrix(R):
    """Normalize rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix with small numerical errors

    Returns
    -------
    R : array, shape (3, 3)
        Normalized rotation matrix
    """
    R = np.asarray(R)
    c2 = R[:, 1]
    c3 = norm_vector(R[:, 2])
    c1 = norm_vector(np.cross(c2, c3))
    c2 = norm_vector(np.cross(c3, c1))
    return np.column_stack((c1, c2, c3))


def norm_angle(a):
    """Normalize angle to (-pi, pi].

    Parameters
    ----------
    a : float or array-like, shape (n,)
        Angle(s) in radians

    Returns
    -------
    a_norm : float or array-like, shape (n,)
        Normalized angle(s) in radians
    """
    # Source of the solution: http://stackoverflow.com/a/32266181
    return -((np.pi - np.asarray(a)) % (2.0 * np.pi) - np.pi)


def norm_axis_angle(a):
    """Normalize axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The length
        of the axis vector is 1 and the angle is in [0, pi). No rotation
        is represented by [1, 0, 0, 0].
    """
    angle = a[3]
    norm = np.linalg.norm(a[:3])
    if angle == 0.0 or norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    res = np.empty(4)
    res[:3] = a[:3] / norm

    angle = norm_angle(angle)
    if angle < 0.0:
        angle *= -1.0
        res[:3] *= -1.0

    res[3] = angle

    return res


def norm_compact_axis_angle(a):
    """Normalize compact axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).
        The angle is in [0, pi). No rotation is represented by [0, 0, 0].
    """
    angle = np.linalg.norm(a)
    if angle == 0.0:
        return np.zeros(3)
    axis = a / angle
    return axis * norm_angle(angle)


def perpendicular_to_vectors(a, b):
    """Compute perpendicular vector to two other vectors.

    Parameters
    ----------
    a : array-like, shape (3,)
        3d vector

    b : array-like, shape (3,)
        3d vector

    Returns
    -------
    c : array-like, shape (3,)
        3d vector that is orthogonal to a and b
    """
    return np.cross(a, b)


def perpendicular_to_vector(a):
    """Compute perpendicular vector to one other vector.

    There is an infinite number of solutions to this problem. Thus, we
    restrict the solutions to [1, 0, z] and return [0, 0, 1] if the
    z component of a is 0.

    Parameters
    ----------
    a : array-like, shape (3,)
        3d vector

    Returns
    -------
    b : array-like, shape (3,)
        A 3d vector that is orthogonal to a. It does not necessarily have
        unit length.
    """
    if abs(a[2]) < eps:
        return np.copy(unitz)
    # Now that we solved the problem for [x, y, 0], we can solve it for all
    # other vectors by restricting solutions to [1, 0, z] and find z.
    # The dot product of orthogonal vectors is 0, thus
    # a[0] * 1 + a[1] * 0 + a[2] * z == 0 or -a[0] / a[2] = z
    return np.array([1.0, 0.0, -a[0] / a[2]])


def angle_between_vectors(a, b, fast=False):
    """Compute angle between two vectors.

    Parameters
    ----------
    a : array-like, shape (n,)
        nd vector

    b : array-like, shape (n,)
        nd vector

    fast : bool, optional (default: False)
        Use fast implementation instead of numerically stable solution

    Returns
    -------
    angle : float
        Angle between a and b
    """
    if len(a) != 3 or fast:
        return np.arccos(
            np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
                    -1.0, 1.0))
    else:
        return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


def vector_projection(a, b):
    """Orthogonal projection of vector a on vector b.

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector a that will be projected on vector b

    b : array-like, shape (3,)
        Vector b on which vector a will be projected

    Returns
    -------
    a_on_b : array, shape (3,)
        Vector a
    """
    b_norm_squared = np.dot(b, b)
    if b_norm_squared == 0.0:
        return np.zeros(3)
    return np.dot(a, b) * b / b_norm_squared


def random_vector(random_state=np.random.RandomState(0), n=3):
    """Generate an nd vector with normally distributed components.

    Each component will be sampled from :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    n : int, optional (default: 3)
        Number of vector components

    Returns
    -------
    v : array-like, shape (n,)
        Random vector
    """
    return random_state.randn(n)


def random_axis_angle(random_state=np.random.RandomState(0)):
    """Generate random axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and than the axis will be normalized
    to length 1.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    """
    angle = np.pi * random_state.rand()
    a = np.array([0, 0, 0, angle])
    a[:3] = norm_vector(random_state.randn(3))
    return a


def random_compact_axis_angle(random_state=np.random.RandomState(0)):
    """Generate random compact axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and than the axis will be normalized
    to length 1.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)
    """
    a = random_axis_angle(random_state)
    return a[:3] * a[3]


def random_quaternion(random_state=np.random.RandomState(0)):
    """Generate random quaternion.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    return norm_vector(random_state.randn(4))


def cross_product_matrix(v):
    """Generate the cross-product matrix of a vector.

    The cross-product matrix :math:`\\boldsymbol{V}` satisfies the equation

    .. math::

        \\boldsymbol{V} \\boldsymbol{w} = \\boldsymbol{v} \\times
        \\boldsymbol{w}

    It is a skew-symmetric (antisymmetric) matrix, i.e.
    :math:`-\\boldsymbol{V} = \\boldsymbol{V}^T`.

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


def check_skew_symmetric_matrix(V, tolerance=1e-6, strict_check=True):
    """Input validation of a skew-symmetric matrix.

    Check whether the transpose of the matrix is its negative:

    .. math::

        V^T = -V

    Parameters
    ----------
    V : array-like, shape (3, 3)
        Cross-product matrix

    tolerance : float, optional (default: 1e-6)
        Tolerance threshold for checks.

    strict_check : bool, optional (default: True)
        Raise a ValueError if V.T is not numerically close enough to -V.
        Otherwise we print a warning.

    Returns
    -------
    V : array-like, shape (3, 3)
        Validated cross-product matrix
    """
    V = np.asarray(V, dtype=np.float)
    if V.ndim != 2 or V.shape[0] != 3 or V.shape[1] != 3:
        raise ValueError("Expected skew-symmetric matrix with shape (3, 3), "
                         "got array-like object with shape %s" % (V.shape,))
    if not np.allclose(V.T, -V, atol=tolerance):
        error_msg = ("Expected skew-symmetric matrix, but it failed the test "
                     "V.T = %r\n-V = %r" % (V.T, -V))
        if strict_check:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg)
    return V


def check_matrix(R, tolerance=1e-6, strict_check=True):
    """Input validation of a rotation matrix.

    We check whether R multiplied by its inverse is approximately the identity
    matrix and the determinant is approximately 1.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    tolerance : float, optional (default: 1e-6)
        Tolerance threshold for checks. Default tolerance is the same as in
        assert_rotation_matrix(R).

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    R : array, shape (3, 3)
        Validated rotation matrix
    """
    R = np.asarray(R, dtype=np.float)
    if R.ndim != 2 or R.shape[0] != 3 or R.shape[1] != 3:
        raise ValueError("Expected rotation matrix with shape (3, 3), got "
                         "array-like object with shape %s" % (R.shape,))
    RRT = np.dot(R, R.T)
    if not np.allclose(RRT, np.eye(3), atol=tolerance):
        error_msg = ("Expected rotation matrix, but it failed the test "
                     "for inversion by transposition. np.dot(R, R.T) "
                     "gives %r" % RRT)
        if strict_check:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg)
    R_det = np.linalg.det(R)
    if abs(R_det - 1) > tolerance:
        error_msg = ("Expected rotation matrix, but it failed the test "
                     "for the determinant, which should be 1 but is %g; "
                     "that is, it probably represents a rotoreflection"
                     % R_det)
        if strict_check:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg)
    return R


def check_axis_angle(a):
    """Input validation of axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array, shape (4,)
        Validated axis of rotation and rotation angle: (x, y, z, angle)
    """
    a = np.asarray(a, dtype=np.float)
    if a.ndim != 1 or a.shape[0] != 4:
        raise ValueError("Expected axis and angle in array with shape (4,), "
                         "got array-like object with shape %s" % (a.shape,))
    return norm_axis_angle(a)


def check_compact_axis_angle(a):
    """Input validation of compact axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    Returns
    -------
    a : array, shape (3,)
        Validated axis of rotation and rotation angle: angle * (x, y, z)
    """
    a = np.asarray(a, dtype=np.float)
    if a.ndim != 1 or a.shape[0] != 3:
        raise ValueError("Expected axis and angle in array with shape (3,), "
                         "got array-like object with shape %s" % (a.shape,))
    return norm_compact_axis_angle(a)


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
    q : array-like, shape (4,)
        Validated quaternion to represent rotation: (w, x, y, z)
    """
    q = np.asarray(q, dtype=np.float)
    if q.ndim != 1 or q.shape[0] != 4:
        raise ValueError("Expected quaternion with shape (4,), got "
                         "array-like object with shape %s" % (q.shape,))
    if unit:
        return norm_vector(q)
    else:
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
    Q : array-like, shape (n_steps, 4)
        Validated quaternions to represent rotations: (w, x, y, z)
    """
    Q_checked = np.asarray(Q, dtype=np.float)
    if Q_checked.ndim != 2 or Q_checked.shape[1] != 4:
        raise ValueError("Expected quaternion array with shape (n_steps, 4), got "
                         "array-like object with shape %s" % (Q_checked.shape,))
    if unit:
        for i in range(len(Q)):
            Q_checked[i] = norm_vector(Q_checked[i])
    return Q_checked


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
        Second vector, must not be 0 or parallel to v1

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
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
    """Compute rotation matrix from axis-angle.

    This is called exponential map or Rodrigues' formula.

    This typically results in an active rotation matrix.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    R : array-like, shape (3, 3)
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

    return R


def matrix_from_compact_axis_angle(a):
    """Compute rotation matrix from compact axis-angle.

    This is called exponential map or Rodrigues' formula.

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


def matrix_from_angle(basis, angle):
    """Compute passive rotation matrix from rotation around basis vector.

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


passive_matrix_from_angle = matrix_from_angle


def active_matrix_from_angle(basis, angle):
    """Compute active rotation matrix from rotation around basis vector.

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


def matrix_from_euler_xyz(e):
    """Compute passive rotation matrix from intrinsic xyz Tait-Bryan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = passive_matrix_from_angle(0, alpha).dot(
        passive_matrix_from_angle(1, beta)).dot(
        passive_matrix_from_angle(2, gamma))
    return R


def matrix_from_euler_zyx(e):
    """Compute passive rotation matrix from intrinsic zyx Tait-Bryan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    gamma, beta, alpha = e
    R = passive_matrix_from_angle(2, gamma).dot(
        passive_matrix_from_angle(1, beta)).dot(
        passive_matrix_from_angle(0, alpha))
    return R


def active_matrix_from_intrinsic_euler_xzx(e):
    """Compute active rotation matrix from intrinsic xzx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
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
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = active_matrix_from_angle(2, alpha).dot(
        active_matrix_from_angle(0, beta)).dot(
        active_matrix_from_angle(1, gamma))
    return R


def active_matrix_from_extrinsic_euler_zxy(e):
    """Compute active rotation matrix from extrinsic zyx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array-like, shape (3, 3)
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
        Angles for rotation around x- (roll), y- (pitch), and z-axes (yaw), extrinsic rotations

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    roll, pitch, yaw = rpy
    R = active_matrix_from_angle(2, yaw).dot(
        active_matrix_from_angle(1, pitch)).dot(
        active_matrix_from_angle(0, roll))
    return R


def matrix_from(R=None, a=None, q=None, e_xyz=None, e_zyx=None):
    """Compute rotation matrix from another representation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    e_xyz : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)

    e_zyx : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    if R is not None:
        return R
    if a is not None:
        return matrix_from_axis_angle(a)
    if q is not None:
        return matrix_from_quaternion(q)
    if e_xyz is not None:
        return matrix_from_euler_xyz(e_xyz)
    if e_zyx is not None:
        return matrix_from_euler_zyx(e_zyx)
    raise ValueError("Cannot compute rotation matrix from no rotation.")


def _general_intrinsic_euler_from_active_matrix(R, n1, n2, n3, proper_euler, strict_check=True):
    """TODO

    https://arc.aiaa.org/doi/abs/10.2514/1.16622
    """
    # TODO https://github.com/scipy/scipy/blob/master/scipy/spatial/transform/rotation.pyx#L89
    D = check_matrix(R, strict_check=strict_check)

    # Step 2
    # - Equation 5
    n1_cross_n2 = np.cross(n1, n2)
    lmbda = np.arctan2(
        np.dot(n1_cross_n2, n3),
        np.dot(n1, n3)
    )
    # - Equation 6
    C = np.vstack((n2, n1_cross_n2, n1))  # same as column_stack(...).T
    CDCT = np.dot(np.dot(C, D), C.T)

    # Step 3
    # - Equation 8
    O = np.dot(CDCT, active_matrix_from_angle(0, lmbda).T)

    # Step 4
    # - Equation 10a
    beta = lmbda + np.arccos(O[2, 2])

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
        # Step 6
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


def euler_xyz_from_matrix(R, strict_check=True):
    """Compute xyz Euler angles from passive rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Passive rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e_xyz : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)
    """
    R = check_matrix(R, strict_check=strict_check)
    if np.abs(R[0, 2]) != 1.0:
        # NOTE: There are two solutions: angle2 and pi - angle2!
        angle2 = np.arcsin(-R[0, 2])
        angle1 = np.arctan2(R[1, 2] / np.cos(angle2), R[2, 2] / np.cos(angle2))
        angle3 = np.arctan2(R[0, 1] / np.cos(angle2), R[0, 0] / np.cos(angle2))
    else:
        if R[0, 2] == 1.0:
            angle3 = 0.0
            angle2 = -np.pi / 2.0
            angle1 = np.arctan2(-R[1, 0], -R[2, 0])
        else:
            angle3 = 0.0
            angle2 = np.pi / 2.0
            angle1 = np.arctan2(R[1, 0], R[2, 0])
    return np.array([angle1, angle2, angle3])


def euler_zyx_from_matrix(R, strict_check=True):
    """Compute zyx Euler angles from passive rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Passive rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e_zyx : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)
    """
    R = check_matrix(R, strict_check=strict_check)
    if np.abs(R[2, 0]) != 1.0:
        # NOTE: There are two solutions: angle2 and pi - angle2!
        angle2 = np.arcsin(R[2, 0])
        angle3 = np.arctan2(-R[2, 1] / np.cos(angle2),
                            R[2, 2] / np.cos(angle2))
        angle1 = np.arctan2(-R[1, 0] / np.cos(angle2),
                            R[0, 0] / np.cos(angle2))
    else:
        if R[2, 0] == 1.0:
            angle3 = 0.0
            angle2 = np.pi / 2.0
            angle1 = np.arctan2(R[0, 1], -R[0, 2])
        else:
            angle3 = 0.0
            angle2 = -np.pi / 2.0
            angle1 = np.arctan2(R[0, 1], R[0, 2])
    return np.array([angle1, angle2, angle3])


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
    e : array-like, shape (3,)
        Angles for rotation around z-, x-, and y-axes (extrinsic rotations)
    """
    return _general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unitz, False, strict_check)[::-1]


def axis_angle_from_matrix(R, strict_check=True):
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

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    R = check_matrix(R, strict_check=strict_check)
    angle = np.arccos((np.trace(R) - 1.0) / 2.0)

    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    a = np.empty(4)

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    axis_unnormalized = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if abs(angle - np.pi) < 1e-4:
        # The threshold is a result from this discussion:
        # https://github.com/rock-learning/pytransform3d/issues/43
        # The standard formula becomes numerically unstable, however,
        # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
        # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
        # squared values of the rotation axis on the diagonal of this matrix.
        # We can still use the original formula to reconstruct the signs of
        # the rotation axis correctly.
        a[:3] = np.sqrt(0.5 * (np.diag(R) + 1.0)) * np.sign(axis_unnormalized)
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
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    q = check_quaternion(q)
    p = q[1:]
    p_norm = np.linalg.norm(p)
    if p_norm < np.finfo(float).eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = p / p_norm
        angle = (2.0 * np.arccos(q[0]),)
        return np.hstack((axis, angle))


def axis_angle_from_compact_axis_angle(a):
    """Compute axis-angle from compact axis-angle representation.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z).

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    a = check_compact_axis_angle(a)
    angle = np.linalg.norm(a)
    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
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
    a : array-like, shape (4,)
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
    aa[3] = np.arccos(cos_angle)
    return norm_axis_angle(aa)


def compact_axis_angle(a):
    """Compute 3-dimensional axis-angle from a 4-dimensional one.

    In a 3-dimensional axis-angle, the 4th dimension (the rotation) is
    represented by the norm of the rotation axis vector, which means we
    transform :math:`\\left( \\boldsymbol{\hat{e}}, \\theta \\right)` to
    :math:`\\theta \\boldsymbol{\hat{e}}`.

    We usually assume active rotations.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle).

    Returns
    -------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z) (compact
        representation).
    """
    a = check_axis_angle(a)
    return a[:3] * a[3]


def compact_axis_angle_from_matrix(R):
    """Compute compact axis-angle from rotation matrix.

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

    Returns
    -------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].
    """
    a = axis_angle_from_matrix(R)
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
    a : array-like, shape (3,)
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
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    R = check_matrix(R, strict_check=strict_check)
    q = np.empty(4)

    # Source: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
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
    q : array-like, shape (4,)
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
    q : array-like, shape (4,)
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
    q_xyzw : array-like, shape (4,)
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
    q_wxyz : array-like, shape (4,)
        Quaternion with scalar part before vector part
    """
    q_xyzw = check_quaternion(q_xyzw)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def quaternion_integrate(Qd, q0=np.array([1.0, 0.0, 0.0, 0.0]), dt=1.0):
    """Integrate angular velocities to quaternions.

    Parameters
    ----------
    A : array-like, shape (n_steps, 3)
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
    quaternions as [NumPy's gradient
    function](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html)
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
    Qd[0] = compact_axis_angle_from_quaternion(concatenate_quaternions(Q[1], q_conj(Q[0]))) / dt
    for t in range(1, len(Q) - 1):
        # divided by two because of central differences
        Qd[t] = compact_axis_angle_from_quaternion(concatenate_quaternions(Q[t + 1], q_conj(Q[t - 1]))) / (2.0 * dt)
    Qd[-1] = compact_axis_angle_from_quaternion(concatenate_quaternions(Q[-1], q_conj(Q[-2]))) / dt
    return Qd


def concatenate_quaternions(q1, q2):
    """Concatenate two quaternions.

    We use Hamilton's quaternion multiplication.

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


def _slerp_weights(angle, t):
    if angle == 0.0:
        return (np.ones_like(t), np.zeros_like(t))
    else:
        return (np.sin((1.0 - t) * angle) / np.sin(angle),
                np.sin(t * angle) / np.sin(angle))


def axis_angle_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start axis of rotation and rotation angle: (x, y, z, angle)

    end : array-like, shape (4,)
        Goal axis of rotation and rotation angle: (x, y, z, angle)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    a : array-like, shape (4,)
        Interpolated axis of rotation and rotation angle: (x, y, z, angle)
    """
    start = check_axis_angle(start)
    end = check_axis_angle(end)
    angle = angle_between_vectors(start[:3], end[:3])
    w1, w2 = _slerp_weights(angle, t)
    w1 = np.array([w1, w1, w1, (1.0 - t)])
    w2 = np.array([w2, w2, w2, t])
    return w1 * start + w2 * end


def quaternion_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start unit quaternion to represent rotation: (w, x, y, z)

    end : array-like, shape (4,)
        End unit quaternion to represent rotation: (w, x, y, z)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    q : array-like, shape (4,)
        Interpolated unit quaternion to represent rotation: (w, x, y, z)
    """
    start = check_quaternion(start)
    end = check_quaternion(end)
    angle = angle_between_vectors(start, end)
    w1, w2 = _slerp_weights(angle, t)
    return w1 * start + w2 * end


def quaternion_dist(q1, q2):
    """Compute distance between two quaternions.

    We use the angular metric of :math:`S^3`, which is defined as

    .. math::

        d(q_1, q_2) = \\min(|| \\log(q_1 * \\overline{q_2})||,
                            2 \\pi - || \\log(q_1 * \\overline{q_2})||)

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
    """Compute the rotation in angle-axis format that rotates q2 into q1.

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


def plot_basis(ax=None, R=None, p=np.zeros(3), s=1.0, ax_s=1, strict_check=True, **kwargs):
    """Plot basis of a rotation matrix.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    R : array-like, shape (3, 3), optional (default: I)
        Rotation matrix, each column contains a basis vector

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the frame that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from .plot_utils import make_3d_axis, Frame
    if ax is None:
        ax = make_3d_axis(ax_s)

    if R is None:
        R = np.eye(3)
    R = check_matrix(R, strict_check=strict_check)

    A2B = np.eye(4)
    A2B[:3, :3] = R
    A2B[:3, 3] = p

    frame = Frame(A2B, s=s, **kwargs)
    frame.add_frame(ax)

    return ax


def plot_axis_angle(ax=None, a=a_id, p=p0, s=1.0, ax_s=1, **kwargs):
    """Plot rotation axis and angle.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    a : array-like, shape (4,), optional (default: [1, 0, 0, 0])
        Axis of rotation and rotation angle: (x, y, z, angle)

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from .plot_utils import make_3d_axis, Arrow3D
    a = check_axis_angle(a)
    if ax is None:
        ax = make_3d_axis(ax_s)

    axis_arrow = Arrow3D(
        [p[0], p[0] + s * a[0]],
        [p[1], p[1] + s * a[1]],
        [p[2], p[2] + s * a[2]],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(axis_arrow)

    p1 = (unitx if np.abs(a[0]) <= np.finfo(float).eps else
          perpendicular_to_vectors(unity, a[:3]))
    p2 = perpendicular_to_vectors(a[:3], p1)

    angle_p1p2 = angle_between_vectors(p1, p2)
    arc = np.empty((100, 3))
    for i, t in enumerate(np.linspace(0, 2 * a[3] / np.pi, len(arc))):
        w1, w2 = _slerp_weights(angle_p1p2, t)
        arc[i] = p + 0.5 * s * (a[:3] + w1 * p1 + w2 * p2)
    ax.plot(arc[:-5, 0], arc[:-5, 1], arc[:-5, 2], color="k", lw=3, **kwargs)

    arrow_coords = np.vstack((arc[-1], arc[-1] + 20 * (arc[-1] - arc[-3]))).T
    angle_arrow = Arrow3D(
        arrow_coords[0], arrow_coords[1], arrow_coords[2],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(angle_arrow)

    for i in [0, -1]:
        arc_bound = np.vstack((p + 0.5 * s * a[:3], arc[i])).T
        ax.plot(arc_bound[0], arc_bound[1], arc_bound[2], "--", c="k")

    return ax


def assert_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the rotation axis is normalized to length 1 and
    the angle is within [0, pi). However, this function ignores these
    constraints and will normalize the representations before comparison.
    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    a1 : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    a2 : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    """
    # required despite normalization in case of 180 degree rotation
    if np.any(np.sign(a1) != np.sign(a2)):
        a1 = -a1
    a1 = norm_axis_angle(a1)
    a2 = norm_axis_angle(a2)
    assert_array_almost_equal(a1, a2, *args, **kwargs)


def assert_compact_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the angle is within [0, pi). However, this function
    ignores this constraint and will normalize the representations before
    comparison. See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    a1 : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    a2 : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)
    """
    angle1 = np.linalg.norm(a1)
    angle2 = np.linalg.norm(a2)
    # required despite normalization in case of 180 degree rotation
    if (abs(angle1) == np.pi and abs(angle2) == np.pi and
        any(np.sign(a1) != np.sign(a2))):
        a1 = -a1
    a1 = norm_compact_axis_angle(a1)
    a2 = norm_compact_axis_angle(a2)
    assert_array_almost_equal(a1, a2, *args, **kwargs)


def assert_quaternion_equal(q1, q2, *args, **kwargs):
    """Raise an assertion if two quaternions are not approximately equal.

    Note that quaternions are equal either if q1 == q2 or if q1 == -q2. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.
    """
    try:
        assert_array_almost_equal(q1, q2, *args, **kwargs)
    except AssertionError:
        assert_array_almost_equal(q1, -q2, *args, **kwargs)


def assert_euler_xyz_equal(e_xyz1, e_xyz2, *args, **kwargs):
    """Raise an assertion if two xyz Euler angles are not approximately equal.

    Note that Euler angles are only unique if we limit them to the intervals
    [-pi, pi], [-pi/2, pi/2], and [-pi, pi] respectively. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.
    """
    R1 = matrix_from_euler_xyz(e_xyz1)
    R2 = matrix_from_euler_xyz(e_xyz2)
    assert_array_almost_equal(R1, R2, *args, **kwargs)


def assert_euler_zyx_equal(e_zyx1, e_zyx2, *args, **kwargs):
    """Raise an assertion if two zyx Euler angles are not approximately equal.

    Note that Euler angles are only unique if we limit them to the intervals
    [-pi, pi], [-pi/2, pi/2], and [-pi, pi] respectively. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.
    """
    R1 = matrix_from_euler_zyx(e_zyx1)
    R2 = matrix_from_euler_zyx(e_zyx2)
    assert_array_almost_equal(R1, R2, *args, **kwargs)


def assert_rotation_matrix(R, *args, **kwargs):
    """Raise an assertion if a matrix is not a rotation matrix.

    The two properties :math:`\\boldsymbol{I} = \\boldsymbol{R R}^T` and
    :math:`det(R) = 1` will be checked. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.
    """
    assert_array_almost_equal(np.dot(R, R.T), np.eye(3), *args, **kwargs)
    assert_array_almost_equal(np.linalg.det(R), 1.0, *args, **kwargs)
