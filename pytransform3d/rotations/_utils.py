"""Utility functions for rotations."""
import warnings
import math
import numpy as np
from ._constants import unitz, eps


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


def plane_basis_from_normal(plane_normal):
    """Compute two basis vectors of a plane from the plane's normal vector.

    Note that there are infinitely many solutions because any rotation of the
    basis vectors about the normal is also a solution. This function
    deterministically picks one of the solutions.

    The two basis vectors of the plane together with the normal form an
    orthonormal basis in 3D space and could be used as columns to form a
    rotation matrix.

    Parameters
    ----------
    plane_normal : array-like, shape (3,)
        Plane normal of unit length.

    Returns
    -------
    x_axis : array, shape (3,)
        x-axis of the plane.

    y_axis : array, shape (3,)
        y-axis of the plane.
    """
    if abs(plane_normal[0]) >= abs(plane_normal[1]):
        # x or z is the largest magnitude component, swap them
        length = math.sqrt(
            plane_normal[0] * plane_normal[0]
            + plane_normal[2] * plane_normal[2])
        x_axis = np.array([-plane_normal[2] / length, 0.0,
                           plane_normal[0] / length])
        y_axis = np.array([
            plane_normal[1] * x_axis[2],
            plane_normal[2] * x_axis[0] - plane_normal[0] * x_axis[2],
            -plane_normal[1] * x_axis[0]])
    else:
        # y or z is the largest magnitude component, swap them
        length = math.sqrt(plane_normal[1] * plane_normal[1]
                           + plane_normal[2] * plane_normal[2])
        x_axis = np.array([0.0, plane_normal[2] / length,
                           -plane_normal[1] / length])
        y_axis = np.array([
            plane_normal[1] * x_axis[2] - plane_normal[2] * x_axis[1],
            -plane_normal[0] * x_axis[2], plane_normal[0] * x_axis[1]])
    return x_axis, y_axis


def random_vector(random_state=np.random.RandomState(0), n=3):
    r"""Generate an nd vector with normally distributed components.

    Each component will be sampled from :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    n : int, optional (default: 3)
        Number of vector components

    Returns
    -------
    v : array, shape (n,)
        Random vector
    """
    return random_state.randn(n)


def random_axis_angle(random_state=np.random.RandomState(0)):
    r"""Generate random axis-angle.

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
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    """
    angle = np.pi * random_state.rand()
    a = np.array([0, 0, 0, angle])
    a[:3] = norm_vector(random_state.randn(3))
    return a


def random_compact_axis_angle(random_state=np.random.RandomState(0)):
    r"""Generate random compact axis-angle.

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
    a : array, shape (3,)
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
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    return norm_vector(random_state.randn(4))


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
    V : array, shape (3, 3)
        Validated cross-product matrix

    Raises
    ------
    ValueError
        If input is invalid
    """
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2 or V.shape[0] != 3 or V.shape[1] != 3:
        raise ValueError("Expected skew-symmetric matrix with shape (3, 3), "
                         "got array-like object with shape %s" % (V.shape,))
    if not np.allclose(V.T, -V, atol=tolerance):
        error_msg = ("Expected skew-symmetric matrix, but it failed the test "
                     "V.T = %r\n-V = %r" % (V.T, -V))
        if strict_check:
            raise ValueError(error_msg)
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

    Raises
    ------
    ValueError
        If input is invalid
    """
    R = np.asarray(R, dtype=np.float64)
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
        warnings.warn(error_msg)
    R_det = np.linalg.det(R)
    if abs(R_det - 1) > tolerance:
        error_msg = ("Expected rotation matrix, but it failed the test "
                     "for the determinant, which should be 1 but is %g; "
                     "that is, it probably represents a rotoreflection"
                     % R_det)
        if strict_check:
            raise ValueError(error_msg)
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

    Raises
    ------
    ValueError
        If input is invalid
    """
    a = np.asarray(a, dtype=np.float64)
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

    Raises
    ------
    ValueError
        If input is invalid
    """
    a = np.asarray(a, dtype=np.float64)
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
    Q : array-like, shape (n_steps, 4)
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


def check_rotor(rotor):
    """Input validation of rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    rotor : array, shape (4,)
        Validated rotor (with unit norm): (a, b_yz, b_zx, b_xy)

    Raises
    ------
    ValueError
        If input is invalid
    """
    rotor = np.asarray(rotor, dtype=np.float64)
    if rotor.ndim != 1 or rotor.shape[0] != 4:
        raise ValueError("Expected rotor with shape (4,), got "
                         "array-like object with shape %s" % (rotor.shape,))
    return norm_vector(rotor)
