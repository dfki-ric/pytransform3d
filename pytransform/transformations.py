import numpy as np
from .rotations import (random_quaternion, random_vector,
                        matrix_from_quaternion, plot_basis,
                        assert_rotation_matrix, check_matrix)
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
    return np.linalg.inv(A2B)


def translate_transform(A2B, p, out=None):
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
    if out is None:
        out = A2B.copy()
    l = len(p)
    out[:l, -1] = p
    return out


def rotate_transform(A2B, R, out=None):
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
    if out is None:
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

    A2B : array-like, shape (4, 4)
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


def plot_transform(ax=None, A2B=None, s=1.0, ax_s=1, **kwargs):
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

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if A2B is None:
        A2B = np.eye(4)
    A2B = check_transform(A2B)
    return plot_basis(ax, A2B[:3, :3], A2B[:3, 3], s, ax_s, **kwargs)


def assert_transform(A2B, *args, **kwargs):
    """Raise an assertion if the transform is not a homogeneous matrix.

    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.
    """
    assert_rotation_matrix(A2B[:3, :3], *args, **kwargs)
    assert_array_almost_equal(A2B[3], np.array([0.0, 0.0, 0.0, 1.0]),
                              *args, **kwargs)
