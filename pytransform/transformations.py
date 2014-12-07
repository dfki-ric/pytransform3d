import numpy as np
from .rotations import plot_basis


def invert_transform(A2B):
    """TODO document me"""
    if A2B.shape != (4, 4):
        raise ValueError("Transformation must have shape (4, 4) but has %s"
                         % A2B.shape)
    return np.linalg.inv(A2B)


def transform_from(R, p):
    """TODO document me"""
    A2B = rotate_transform(np.eye(4), R)
    A2B = translate_transform(A2B, p)
    return A2B


def translate_transform(A2B, p, out=None):
    """TODO document me"""
    if out is None:
        out = A2B.copy()
    l = len(p)
    out[:l, -1] = p
    return out


def rotate_transform(A2B, R, out=None):
    """TODO document me"""
    if out is None:
        out = A2B.copy()
    out[:3, :3] = R
    return out


def vector_to_point(v):
    """TODO document me"""
    return np.hstack((v, 1))


def transform(A2B, PB):
    """TODO document me"""
    PB = np.asarray(PB)
    if PB.ndim == 1:
        return np.dot(A2B, PB)
    elif PB.ndim == 2:
        return np.dot(PB, A2B.T)
    else:
        raise ValueError("Cannot transform array with more than 2 dimensions")


def plot_transform(ax=None, A2B=np.eye(4), s=1.0, ax_s=1, **kwargs):
    """TODO document me"""
    return plot_basis(ax, A2B[:3, :3], A2B[:3, 3], s, ax_s, **kwargs)
