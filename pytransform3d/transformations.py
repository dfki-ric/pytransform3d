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


def screw_axis_from_screw_parameters(q, s_axis, h, theta=1.0):
    """TODO

    Parameters
    ----------
    TODO

    Returns
    -------
    twist : TODO
    """
    if np.isinf(h):
        raise NotImplementedError("TODO only translation")
    else:
        return np.r_[s_axis, np.cross(q, s_axis) + h * s_axis] * theta


def transform_from_exponential_coordinates(Stheta):
    """Conversion from twist displacement to homogeneous matrix.

    Exponential map.

    Parameters
    ----------
    TODO

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    omega_theta = Stheta[:3]
    theta = np.linalg.norm(omega_theta)

    if theta == 0.0:  # only translation
        A2B = np.eye(4)
        A2B[:3, 3] = Stheta[3:]
        return A2B

    screw_axis = Stheta / theta
    omega = screw_axis[:3]
    v = screw_axis[3:]

    A2B = np.eye(4)
    A2B[:3, :3] = matrix_from_axis_angle(np.r_[omega, theta])
    omega_matrix = cross_product_matrix(omega)
    A2B[:3, 3] = np.dot(
        np.eye(3) * theta
        + (1.0 - math.cos(theta)) * omega_matrix
        + (theta - math.sin(theta)) * np.dot(omega_matrix, omega_matrix),
        v)
    return A2B


def plot_screw(ax=None, q=np.zeros(3), s_axis=np.array([1.0, 0.0, 0.0]), h=1.0, theta=1.0, ax_s=1, **kwargs):
    """TODO

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    q : TODO

    s_axis : TODO

    h : TODO

    theta : TODO

    # TODO plot in base coordinate frame A2B!!!

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
    from .rotations import unitx, unity, perpendicular_to_vectors, angle_between_vectors, _slerp_weights
    if ax is None:
        ax = make_3d_axis(ax_s)

    ax.scatter(q[0], q[1], q[2], color="r")

    axis_arrow = Arrow3D(
        [q[0], q[0] + s_axis[0]],
        [q[1], q[1] + s_axis[1]],
        [q[2], q[2] + s_axis[2]],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(axis_arrow)

    q_projected_on_s_axis = np.linalg.norm(q) * np.cos(angle_between_vectors(-q, s_axis)) * s_axis
    p1 = -q + -q_projected_on_s_axis
    p2 = perpendicular_to_vectors(s_axis, p1)

    arc = np.empty((40, 3))
    for i, theta_fraction in enumerate(np.linspace(0.0, theta, len(arc))):
        w1, w2 = _slerp_weights(theta, theta_fraction / theta)
        arc[i] = q + q_projected_on_s_axis + w1 * p1 + w2 * p2 + h * s_axis * theta_fraction
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", lw=3, **kwargs)

    arrow_coords = np.vstack((arc[-1], arc[-1] + (arc[-1] - arc[-2]))).T
    angle_arrow = Arrow3D(
        arrow_coords[0], arrow_coords[1], arrow_coords[2],
        mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
    ax.add_artist(angle_arrow)

    for i in [0, -1]:
        arc_bound = np.vstack((q, arc[i])).T
        ax.plot(arc_bound[0], arc_bound[1], arc_bound[2], "--", c="k")
