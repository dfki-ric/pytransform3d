"""Transformation matrices."""
import warnings
import numpy as np
from ..rotations import matrix_requires_renormalization, check_matrix


def transform_requires_renormalization(A2B, tolerance=1e-6):
    r"""Check if transformation matrix requires renormalization.

    This function will check if :math:`R R^T \approx I`.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B with a rotation matrix that should
        be orthonormal.

    tolerance : float, optional (default: 1e-6)
        Tolerance for check.

    Returns
    -------
    required : bool
        Indicates if renormalization is required.

    See Also
    --------
    pytransform3d.rotations.matrix_requires_renormalization
        Check if a rotation matrix needs renormalization.
    pytransform3d.rotations.norm_matrix : Orthonormalize rotation matrix.
    """
    return matrix_requires_renormalization(np.asarray(A2B[:3, :3]), tolerance)


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

    Raises
    ------
    ValueError
        If input is invalid
    """
    A2B = np.asarray(A2B, dtype=np.float64)
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
        warnings.warn(error_msg)
    return A2B
