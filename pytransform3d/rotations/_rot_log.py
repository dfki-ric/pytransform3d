"""Conversions between rotation representations."""
import warnings

import numpy as np


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


check_rot_log = check_skew_symmetric_matrix


def cross_product_matrix(v):
    r"""Generate the cross-product matrix of a vector.

    The cross-product matrix :math:`\boldsymbol{V}` satisfies the equation

    .. math::

        \boldsymbol{V} \boldsymbol{w} = \boldsymbol{v} \times
        \boldsymbol{w}.

    It is a skew-symmetric (antisymmetric) matrix, i.e.,
    :math:`-\boldsymbol{V} = \boldsymbol{V}^T`. Its elements are

    .. math::

        \left[\boldsymbol{v}\right]
        =
        \left[\begin{array}{c}
        v_1\\ v_2\\ v_3
        \end{array}\right]
        =
        \boldsymbol{V}
        =
        \left(\begin{array}{ccc}
        0 & -v_3 & v_2\\
        v_3 & 0 & -v_1\\
        -v_2 & v_1 & 0
        \end{array}\right).

    The function can also be used to compute the logarithm of rotation from
    a compact axis-angle representation.

    Parameters
    ----------
    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    V : array, shape (3, 3)
        Cross-product matrix
    """
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


rot_log_from_compact_axis_angle = cross_product_matrix
