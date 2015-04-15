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
    """TODO document me"""
    return np.hstack((v, 1))


def transform(A2B, PA):
    """TODO document me"""
    PA = np.asarray(PA)
    if PA.ndim == 1:
        return np.dot(A2B, PA)
    elif PA.ndim == 2:
        return np.dot(PA, A2B.T)
    else:
        raise ValueError("Cannot transform array with more than 2 dimensions")


def plot_transform(ax=None, A2B=np.eye(4), s=1.0, ax_s=1, **kwargs):
    """TODO document me"""
    return plot_basis(ax, A2B[:3, :3], A2B[:3, 3], s, ax_s, **kwargs)
