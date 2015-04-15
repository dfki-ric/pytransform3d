import numpy as np
from .rotations import plot_basis


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
    if A2B.shape != (4, 4):
        raise ValueError("Transformation must have shape (4, 4) but has %s"
                         % A2B.shape)
    return np.linalg.inv(A2B)


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


def translate_transform(A2B, p, out=None):
    """Translate transform.

    Parameters
    ----------
    p : array-like, shape (3,)
        Translation

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
    if out is None:
        out = A2B.copy()
    l = len(p)
    out[:l, -1] = p
    return out


def rotate_transform(A2B, R, out=None):
    """Rotate transform.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B
    """
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
    """
    if A2B is None:
        A2B = np.eye(4)
    return plot_basis(ax, A2B[:3, :3], A2B[:3, 3], s, ax_s, **kwargs)
