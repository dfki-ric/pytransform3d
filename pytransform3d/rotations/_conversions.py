"""Conversions between rotation representations."""
import numpy as np


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
