"""Transformations in three dimensions - SE(3)."""
import numpy as np
from .rotations import (random_quaternion, random_vector,
                        matrix_from_quaternion, quaternion_from_matrix,
                        assert_rotation_matrix, check_matrix,
                        norm_vector, axis_angle_from_matrix,
                        matrix_from_axis_angle)
from .plot_utils import Frame, make_3d_axis
from numpy.testing import assert_array_almost_equal


def check_transform(A2B):
    """Input validation of transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

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
    check_matrix(A2B[:3, :3])
    if not np.allclose(A2B[3], np.array([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("Excpected homogeneous transformation matrix with "
                         "[0, 0, 0, 1] at the bottom, got %r" % A2B)
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


def transform_from(R, p):
    """Make transformation from rotation matrix and translation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    p : array-like, shape (3,)
        Translation

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = rotate_transform(np.eye(4), R)
    A2B = translate_transform(A2B, p)
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


def invert_transform(A2B):
    """Invert transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    B2A : array-like, shape (4, 4)
        Transform from frame B to frame A
    """
    A2B = check_transform(A2B)
    # NOTE there is a faster version:
    # ( R t )^-1   ( R^T -R^T*t )
    # ( 0 1 )    = ( 0    1     )
    return np.linalg.inv(A2B)


def translate_transform(A2B, p):
    """Translate transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    p : array-like, shape (3,)
        Translation

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = check_transform(A2B)
    out = A2B.copy()
    l = len(p)
    out[:l, -1] = p
    return out


def rotate_transform(A2B, R):
    """Rotate transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = check_transform(A2B)
    out = A2B.copy()
    out[:3, :3] = R
    return out


def vector_to_point(v):
    """Convert 3D vector to position.

    Parameters
    ----------
    v : array-like, shape (3,)
        3D vector

    Returns
    -------
    p : array-like, shape (4,)
        Point vector with 1 as last element
    """
    return np.hstack((v, 1))


def concat(A2B, B2C):
    """Concatenate transforms.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    B2C : array-like, shape (4, 4)
        Transform from frame B to frame C
    """
    A2B = check_transform(A2B)
    B2C = check_transform(B2C)
    return B2C.dot(A2B)


def transform(A2B, PA):
    """Transform point or list of points.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    PA : array-like, shape (4,) or (n_points, 4)
        Point or points in frame A

    Returns
    -------
    PB : array-like, shape (4,) or (n_points, 4)
        Point or points in frame B
    """
    A2B = check_transform(A2B)
    PA = np.asarray(PA)
    if PA.ndim == 1:
        return np.dot(A2B, PA)
    elif PA.ndim == 2:
        return np.dot(PA, A2B.T)
    else:
        raise ValueError("Cannot transform array with more than 2 dimensions")


def scale_transform(A2B, s_xr=1.0, s_yr=1.0, s_zr=1.0, s_r=1.0,
                    s_xt=1.0, s_yt=1.0, s_zt=1.0, s_t=1.0, s_d=1.0):
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

    Returns
    -------
    A2B_scaled
        Scaled transform from frame A to frame B (actually this is a transform
        from A to another frame C)
    """
    A2B = check_transform(A2B)
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


def pq_from_transform(A2B):
    """Conversion from homogeneous matrix to position and quaternion.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    pq : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    A2B = check_transform(A2B)
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


def plot_transform(ax=None, A2B=None, s=1.0, ax_s=1, name=None, **kwargs):
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

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    if A2B is None:
        A2B = np.eye(4)
    A2B = check_transform(A2B)

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
