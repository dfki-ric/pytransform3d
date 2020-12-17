"""Transformations in three dimensions - SE(3)."""
import warnings
import math
import numpy as np
from .rotations import (random_quaternion, random_vector,
                        matrix_from_quaternion, quaternion_from_matrix,
                        assert_rotation_matrix, check_matrix,
                        norm_vector, axis_angle_from_matrix,
                        matrix_from_axis_angle, cross_product_matrix)
from numpy.testing import assert_array_almost_equal


def check_transform(A2B, strict_check=True):
    """Input validation of transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (4, 4)
        Validated transform from frame A to frame B
    """
    A2B = np.asarray(A2B, dtype=np.float)
    if A2B.ndim != 2 or A2B.shape[0] != 4 or A2B.shape[1] != 4:
        raise ValueError("Expected homogeneous transformation matrix with "
                         "shape (4, 4), got array-like object with shape %s"
                         % (A2B.shape,))
    check_matrix(A2B[:3, :3], strict_check=strict_check)
    if not np.allclose(A2B[3], np.array([0.0, 0.0, 0.0, 1.0])):
        error_msg = ("Excpected homogeneous transformation matrix with "
                     "[0, 0, 0, 1] at the bottom, got %r" % A2B)
        if strict_check:
            raise ValueError(error_msg)
        else:
            warnings.warn(error_msg)
    return A2B


def check_pq(pq):
    """Input validation for position and orientation quaternion.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    pq : array, shape (7,)
        Validated position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    pq = np.asarray(pq, dtype=np.float)
    if pq.ndim != 1 or pq.shape[0] != 7:
        raise ValueError("Expected position and orientation quaternion in a "
                         "1D array, got array-like object with shape %s"
                         % (pq.shape,))
    return pq


def transform_from(R, p, strict_check=True):
    """Make transformation from rotation matrix and translation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = rotate_transform(
        np.eye(4), R, strict_check=strict_check, check=False)
    A2B = translate_transform(
        A2B, p, strict_check=strict_check, check=False)
    return A2B


def random_transform(random_state=np.random.RandomState(0)):
    """Generate an random transform.

    Each component of the translation will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Random transform from frame A to frame B
    """
    q = random_quaternion(random_state)
    R = matrix_from_quaternion(q)
    p = random_vector(random_state, n=3)
    return transform_from(R=R, p=p)


def invert_transform(A2B, strict_check=True, check=True):
    """Invert transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    B2A : array-like, shape (4, 4)
        Transform from frame B to frame A
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    # NOTE there is a faster version, but it is not faster than matrix
    # inversion with numpy:
    # ( R t )^-1   ( R^T -R^T*t )
    # ( 0 1 )    = ( 0    1     )
    return np.linalg.inv(A2B)


def translate_transform(A2B, p, strict_check=True, check=True):
    """Sets the translation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    l = len(p)
    out[:l, -1] = p
    return out


def rotate_transform(A2B, R, strict_check=True, check=True):
    """Sets the rotation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, :3] = R
    return out


def vector_to_point(v):
    """Convert 3D vector to position.

    A point (x, y, z) given by the components of a vector will be represented
    by [x, y, z, 1] in homogeneous coordinates to which we can apply a
    transformation.

    Parameters
    ----------
    v : array-like, shape (3,)
        3D vector that contains x, y, and z

    Returns
    -------
    p : array-like, shape (4,)
        Point vector with 1 as last element
    """
    return np.hstack((v, 1))


def vectors_to_points(V):
    """Convert 3D vectors to positions.

    A point (x, y, z) given by the components of a vector will be represented
    by [x, y, z, 1] in homogeneous coordinates to which we can apply a
    transformation.

    Parameters
    ----------
    V : array-like, shape (n_points, 3)
        Each row is a 3D vector that contains x, y, and z

    Returns
    -------
    P : array-like, shape (n_points, 4)
        Each row is a point vector with 1 as last element
    """
    return np.hstack((V, np.ones((len(V), 1))))


def vector_to_direction(v):
    """Convert 3D vector to direction.

    A direction (x, y, z) given by the components of a vector will be
    represented by [x, y, z, 0] in homogeneous coordinates to which we can
    apply a transformation.

    Parameters
    ----------
    v : array-like, shape (3,)
        3D vector that contains x, y, and z

    Returns
    -------
    p : array-like, shape (4,)
        Direction vector with 0 as last element
    """
    return np.hstack((v, 0))


def vectors_to_directions(V):
    """Convert 3D vectors to directions.

    A direction (x, y, z) given by the components of a vector will be
    represented by [x, y, z, 0] in homogeneous coordinates to which we can
    apply a transformation.

    Parameters
    ----------
    V : array-like, shape (n_directions, 3)
        Each row is a 3D vector that contains x, y, and z

    Returns
    -------
    P : array-like, shape (n_directions, 4)
        Each row is a direction vector with 0 as last element
    """
    return np.hstack((V, np.zeros((len(V), 1))))


def concat(A2B, B2C, strict_check=True, check=True):
    """Concatenate transforms.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    B2C : array-like, shape (4, 4)
        Transform from frame B to frame C

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrices are valid

    Returns
    -------
    A2C : array-like, shape (4, 4)
        Transform from frame A to frame C
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
        B2C = check_transform(B2C, strict_check=strict_check)
    return B2C.dot(A2B)


def transform(A2B, PA, strict_check=True):
    """Transform point or list of points or directions.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    PA : array-like, shape (4,) or (n_points, 4)
        Point or points in frame A

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    PB : array-like, shape (4,) or (n_points, 4)
        Point or points in frame B
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    PA = np.asarray(PA)
    if PA.ndim == 1:
        return np.dot(A2B, PA)
    elif PA.ndim == 2:
        return np.dot(PA, A2B.T)
    else:
        raise ValueError("Cannot transform array with more than 2 dimensions")


def scale_transform(A2B, s_xr=1.0, s_yr=1.0, s_zr=1.0, s_r=1.0,
                    s_xt=1.0, s_yt=1.0, s_zt=1.0, s_t=1.0, s_d=1.0,
                    strict_check=True):
    """Scale a transform from A to reference frame B.

    See algorithm 10 from "Analytic Approaches for Design and Operation of
    Haptic Human-Machine Interfaces" (Bertold Bongardt).

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    s_xr : float, optional (default: 1)
        Scaling of x-component of the rotation axis

    s_yr : float, optional (default: 1)
        Scaling of y-component of the rotation axis

    s_zr : float, optional (default: 1)
        Scaling of z-component of the rotation axis

    s_r : float, optional (default: 1)
        Scaling of the rotation

    s_xt : float, optional (default: 1)
        Scaling of z-component of the translation

    s_yt : float, optional (default: 1)
        Scaling of z-component of the translation

    s_zt : float, optional (default: 1)
        Scaling of z-component of the translation

    s_t : float, optional (default: 1)
        Scaling of the translation

    s_d : float, optional (default: 1)
        Scaling of the whole transform (displacement)

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B_scaled
        Scaled transform from frame A to frame B (actually this is a transform
        from A to another frame C)
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    A2B_scaled = np.eye(4)

    R = A2B[:3, :3]
    t = A2B[:3, 3]

    S_t = np.array([s_xt, s_yt, s_zt])
    A2B_scaled[:3, 3] = s_d * s_t * S_t * t

    a = axis_angle_from_matrix(R)
    a_new = np.empty(4)
    a_new[3] = s_d * s_r * a[3]
    S_r = np.array([s_xr, s_yr, s_zr])
    a_new[:3] = norm_vector(S_r * a[:3])
    A2B_scaled[:3, :3] = matrix_from_axis_angle(a_new)

    return A2B_scaled


def pq_from_transform(A2B, strict_check=True):
    """Conversion from homogeneous matrix to position and quaternion.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    return np.hstack((A2B[:3, 3], quaternion_from_matrix(A2B[:3, :3])))


def transform_from_pq(pq):
    """Conversion from position and quaternion to homogeneous matrix.

    Parameters
    ----------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    pq = check_pq(pq)
    return transform_from(matrix_from_quaternion(pq[3:]), pq[:3])


def plot_transform(ax=None, A2B=None, s=1.0, ax_s=1, name=None, strict_check=True, **kwargs):
    """Plot transform.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    A2B : array-like, shape (4, 4), optional (default: I)
        Transform from frame A to frame B

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    name : string, optional (default: None)
        Name of the frame, will be used for annotation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

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

    if A2B is None:
        A2B = np.eye(4)
    A2B = check_transform(A2B, strict_check=strict_check)

    frame = Frame(A2B, name, s, **kwargs)
    frame.add_frame(ax)

    return ax


def assert_transform(A2B, *args, **kwargs):
    """Raise an assertion if the transform is not a homogeneous matrix.

    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.
    """
    assert_rotation_matrix(A2B[:3, :3], *args, **kwargs)
    assert_array_almost_equal(A2B[3], np.array([0.0, 0.0, 0.0, 1.0]),
                              *args, **kwargs)


def check_screw_parameters(q, s_axis, h):
    """Input validation of screw parameters.

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
    q : array-like, shape (3,)
        Vector to a point on the screw axis. Will be set to zero vector when
        pitch is infinite (pure translation).

    s_axis : array-like, shape (3,)
        Unit direction vector of the screw axis

    h : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.
    """
    s_axis = np.asarray(s_axis, dtype=np.float)
    if s_axis.ndim != 1 or s_axis.shape[0] != 3:
        raise ValueError("Expected 3D vector with shape (3,), got array-like "
                         "object with shape %s" % (s_axis.shape,))
    if np.linalg.norm(s_axis) == 0.0:
        raise ValueError("s_axis must not have norm 0")

    q = np.asarray(q, dtype=np.float)
    if q.ndim != 1 or q.shape[0] != 3:
        raise ValueError("Expected 3D vector with shape (3,), got array-like "
                         "object with shape %s" % (q.shape,))
    if np.isinf(h):  # pure translation
        q = np.zeros(3)

    return q, norm_vector(s_axis), h


def check_screw_axis(screw_axis):
    """Input validation of screw parameters.

    Parameters
    ----------
    screw_axis : array-like, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.

    Returns
    -------
    screw_axis : array-like, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.
    """
    screw_axis = np.asarray(screw_axis, dtype=np.float)
    if screw_axis.ndim != 1 or screw_axis.shape[0] != 6:
        raise ValueError("Expected 3D vector with shape (6,), got array-like "
                         "object with shape %s" % (screw_axis.shape,))

    omega_norm = np.linalg.norm(screw_axis[:3])
    if (abs(omega_norm - 1.0) > np.finfo(float).eps
            and abs(omega_norm) > np.finfo(float).eps):
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

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation.

    Returns
    -------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation.
    """
    Stheta = np.asarray(Stheta, dtype=np.float)
    if Stheta.ndim != 1 or Stheta.shape[0] != 6:
        raise ValueError("Expected array-like with shape (6,), got array-like "
                         "object with shape %s" % (Stheta.shape,))
    return Stheta


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
    q : array-like, shape (3,)
        Vector to a point on the screw axis that is orthogonal to s_axis

    s_axis : array-like, shape (3,)
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
        q_cross_s_axis = v - h * s_axis
        q = np.cross(s_axis, q_cross_s_axis)
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
    else:
        return np.r_[s_axis, np.cross(q, s_axis) + h * s_axis]


def screw_axis_from_exponential_coordinates(Stheta):
    Stheta = check_exponential_coordinates(Stheta)
    omega_theta = Stheta[:3]
    v_theta = Stheta[3:]
    theta = np.linalg.norm(omega_theta)
    if theta == 0.0:
        theta = np.linalg.norm(v_theta)
    return Stheta / theta


def screw_axis_from_unit_twist(unit_twist):
    # TODO check unit twist
    screw_axis = np.empty(6)
    screw_axis[0] = unit_twist[2, 1]
    screw_axis[1] = unit_twist[0, 2]
    screw_axis[2] = unit_twist[1, 0]
    screw_axis[3:] = unit_twist[:3, 3]
    return screw_axis


def exponential_coordinates_from_screw_axis(screw_axis, theta):
    return screw_axis * theta


def exponential_coordinates_from_transform_log(transform_log):
    # TODO check matrix log
    Stheta = np.empty(6)
    Stheta[0] = transform_log[2, 1]
    Stheta[1] = transform_log[0, 2]
    Stheta[2] = transform_log[1, 0]
    Stheta[3:] = transform_log[:3, 3]
    return Stheta


def exponential_coordinates_from_transform(A2B, strict_check=True):
    """Conversion from homogeneous matrix to exponential coordinates.

    Logarithmic map.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation.
    """
    A2B = check_transform(A2B, strict_check=strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    if np.linalg.norm(np.eye(3) - R) < np.finfo(float).eps:
        return np.r_[0.0, 0.0, 0.0, p]

    omega_theta = axis_angle_from_matrix(R)
    omega_unit = omega_theta[:3]
    theta = omega_theta[3]
    omega_unit_matrix = cross_product_matrix(omega_unit)

    G_inv = (np.eye(3) / theta - 0.5 * omega_unit_matrix
             + (1.0 / theta - 0.5 / np.tan(theta / 2.0))
             * np.dot(omega_unit_matrix, omega_unit_matrix))
    v = G_inv.dot(p)

    return np.hstack((omega_unit, v)) * theta


def unit_twist_from_screw_axis(screw_axis):
    screw_axis = check_screw_axis(screw_axis)
    omega = screw_axis[:3]
    v = screw_axis[3:]
    unit_twist = np.zeros((4, 4))
    unit_twist[:3, :3] = cross_product_matrix(omega)
    unit_twist[:3, 3] = v
    raise unit_twist


def unit_twist_from_transform_log(transform_log):
    raise NotImplementedError("TODO")


def transform_log_from_exponential_coordinates(Stheta):
    raise NotImplementedError("TODO")


def transform_log_from_unit_twist(unit_twist):
    raise NotImplementedError("TODO")


def transform_log_from_transform(A2B):
    raise NotImplementedError("TODO")


def transform_from_exponential_coordinates(Stheta):
    """Conversion from exponential coordinates to homogeneous matrix.

    Exponential map.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation.

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    Stheta = check_exponential_coordinates(Stheta)
    omega_theta = Stheta[:3]
    theta = np.linalg.norm(omega_theta)

    if theta == 0.0:  # only translation
        return translate_transform(np.eye(4), Stheta[3:])

    screw_axis = Stheta / theta
    omega_unit = screw_axis[:3]
    v = screw_axis[3:]

    A2B = np.eye(4)
    A2B[:3, :3] = matrix_from_axis_angle(np.r_[omega_unit, theta])
    omega_matrix = cross_product_matrix(omega_unit)
    A2B[:3, 3] = np.dot(
        np.eye(3) * theta
        + (1.0 - math.cos(theta)) * omega_matrix
        + (theta - math.sin(theta)) * np.dot(omega_matrix, omega_matrix),
        v)
    return A2B


def transform_from_transform_log(transform_log):
    raise NotImplementedError("TODO")


def plot_screw(ax=None, q=np.zeros(3), s_axis=np.array([1.0, 0.0, 0.0]), h=1.0, theta=1.0, A2B=None, s=1.0, ax_s=1, alpha=1.0, **kwargs):
    """Plot screw axis and transformation.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    q : array-like, shape (3,), optional (default: [0, 0, 0])
        Vector to a point on the screw axis

    s_axis : array-like, shape (3,), optional (default: [1, 0, 0])
        Direction vector of the screw axis

    h : float, optional (default: 1)
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta : float, optional (default: 1)
        Rotation angle. h * theta is the translation.

    A2B : array-like, shape (4, 4), optional (default: I)
        Origin of the screw

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    alpha : float, optional (default: 1)
        Alpha channel of plotted lines

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. color

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    from .plot_utils import make_3d_axis, Arrow3D
    from .rotations import (vector_projection, angle_between_vectors,
                            perpendicular_to_vectors, _slerp_weights)

    if ax is None:
        ax = make_3d_axis(ax_s)

    if A2B is None:
        A2B = np.eye(4)

    q, s_axis, h = check_screw_parameters(q, s_axis, h)

    origin_projected_on_screw_axis = q + vector_projection(-q, s_axis)

    pure_translation = np.isinf(h)

    if not pure_translation:
        screw_axis_to_old_frame = -origin_projected_on_screw_axis
        screw_axis_to_rotated_frame = perpendicular_to_vectors(s_axis, screw_axis_to_old_frame)
        screw_axis_to_translated_frame = h * s_axis

        arc = np.empty((100, 3))
        angle = angle_between_vectors(screw_axis_to_old_frame, screw_axis_to_rotated_frame)
        for i, t in enumerate(zip(np.linspace(0, 2 * theta / np.pi, len(arc)), np.linspace(0.0, 1.0, len(arc)))):
            t1, t2 = t
            w1, w2 = _slerp_weights(angle, t1)
            arc[i] = (origin_projected_on_screw_axis
                      + w1 * screw_axis_to_old_frame
                      + w2 * screw_axis_to_rotated_frame
                      + screw_axis_to_translated_frame * t2 * theta)

    q = transform(A2B, vector_to_point(q))[:3]
    s_axis = transform(A2B, vector_to_direction(s_axis))[:3]
    if not pure_translation:
        arc = transform(A2B, vectors_to_points(arc))[:, :3]
        origin_projected_on_screw_axis = transform(A2B, vector_to_point(origin_projected_on_screw_axis))[:3]

    # Screw axis
    ax.scatter(q[0], q[1], q[2], color="r")
    if pure_translation:
        s_axis *= theta
        ax.scatter(q[0] + s_axis[0], q[1] + s_axis[1], q[2] + s_axis[2], color="r")
    ax.plot(
        [q[0] - s * s_axis[0], q[0] + (1 + s) * s_axis[0]],
        [q[1] - s * s_axis[1], q[1] + (1 + s) * s_axis[1]],
        [q[2] - s * s_axis[2], q[2] + (1 + s) * s_axis[2]],
        "--", c="k", alpha=alpha)
    axis_arrow = Arrow3D(
        [q[0], q[0] + s_axis[0]],
        [q[1], q[1] + s_axis[1]],
        [q[2], q[2] + s_axis[2]],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k", alpha=alpha)
    ax.add_artist(axis_arrow)

    if not pure_translation:
        # Transformation
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", lw=3,
                alpha=alpha, **kwargs)
        arrow_coords = np.vstack((arc[-1], arc[-1] + (arc[-1] - arc[-2]))).T
        angle_arrow = Arrow3D(
            arrow_coords[0], arrow_coords[1], arrow_coords[2],
            mutation_scale=20, lw=3, arrowstyle="-|>", color="k", alpha=alpha)
        ax.add_artist(angle_arrow)

        for i in [0, -1]:
            arc_bound = np.vstack((origin_projected_on_screw_axis, arc[i])).T
            ax.plot(arc_bound[0], arc_bound[1], arc_bound[2], "--", c="k", alpha=alpha)

    return ax
