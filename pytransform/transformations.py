import numpy as np
from .rotations import plot_basis


def invert_transform(A2B):
    """TODO document me"""
    if A2B.shape != (4, 4):
        raise ValueError("Transformation must have shape (4, 4) but has %s"
                         % A2B.shape)
    inv = np.empty((4, 4))
    inv[-1, :] = np.array([0, 0, 0, 1])
    inv[:3, :3] = A2B[:3, :3].T
    inv[:3, -1] = -A2B[:3, -1]
    return inv


def transform_from(R, p):
    """TODO document me"""
    A2B = rotate_transform(np.eye(4), R)
    A2B = translate_transform(A2B, p)
    return A2B


def translate_transform(A2B, p):
    """TODO document me"""
    A2B = A2B.copy()
    l = len(p)
    A2B[:l, -1] = p
    return A2B


def rotate_transform(A2B, R):
    """TODO document me"""
    A2B = A2B.copy()
    A2B[:3, :3] = R
    return A2B


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
