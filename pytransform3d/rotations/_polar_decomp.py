import numpy as np
from ._axis_angle import matrix_from_compact_axis_angle


def robust_polar_decomposition(A, n_iter=20, eps=np.finfo(float).eps):
    r"""Orthonormalize rotation matrix with robust polar decomposition.

    Robust polar decomposition [1]_ [2]_ is a computationally more costly
    method, but it spreads the error more evenly between the basis vectors
    in comparison to Gram-Schmidt orthonormalization (as in
    :func:`norm_matrix`).

    Robust polar decomposition finds an orthonormal matrix that minimizes the
    Frobenius norm

    .. math::

        ||\boldsymbol{A} - \boldsymbol{R}||^2

    between the input :math:`\boldsymbol{A}` that is not orthonormal and the
    output :math:`\boldsymbol{R}` that is orthonormal.

    Parameters
    ----------
    A : array-like, shape (3, 3)
        Matrix that contains a basis vector in each column. The basis does not
        have to be orthonormal.

    n_iter : int, optional (default: 20)
        Maximum number of iterations for which we refine the estimation of the
        rotation matrix.

    eps : float, optional (default: np.finfo(float).eps)
        Precision for termination criterion of iterative refinement.

    Returns
    -------
    R : array, shape (3, 3)
        Orthonormalized rotation matrix.

    See Also
    --------
    norm_matrix
        The cheaper default orthonormalization method that uses Gram-Schmidt
        orthonormalization optimized for 3 dimensions.

    References
    ----------
    .. [1] Selstad, J. (2019). Orthonormalization.
       https://zalo.github.io/blog/polar-decomposition/

    .. [2] Müller, M., Bender, J., Chentanez, N., Macklin, M. (2016).
       A Robust Method to Extract the Rotational Part of Deformations.
       In MIG '16: Proceedings of the 9th International Conference on Motion in
       Games, pp. 55-60, doi: 10.1145/2994258.2994269.
    """
    current_R = np.eye(3)
    for _ in range(n_iter):
        column_vector_cross_products = np.cross(
            current_R, A, axisa=0, axisb=0, axisc=1)
        column_vector_dot_products_sum = np.sum(current_R * A)
        omega = (column_vector_cross_products.sum(axis=0)
                 / (abs(column_vector_dot_products_sum) + eps))
        if np.linalg.norm(omega) < eps:
            break
        current_R = np.dot(matrix_from_compact_axis_angle(omega), current_R)
    return current_R
