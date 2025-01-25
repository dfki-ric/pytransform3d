import numpy as np
from ._conversions import (matrix_from_quaternion,
                           quaternion_from_compact_axis_angle)
from ._quaternions import concatenate_quaternions


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
    current_q = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(n_iter):
        current_R = matrix_from_quaternion(current_q)
        omega = ((np.cross(current_R[:, 0], A[:, 0])
                  + np.cross(current_R[:, 1], A[:, 1])
                  + np.cross(current_R[:, 2], A[:, 2]))
                 /
                 (abs(np.dot(current_R[:, 0], A[:, 0])
                      + np.dot(current_R[:, 1], A[:, 1])
                      + np.dot(current_R[:, 2], A[:, 2]))
                  + eps))
        if np.linalg.norm(omega) < eps:
            break
        current_q = concatenate_quaternions(
            quaternion_from_compact_axis_angle(omega), current_q)
    return matrix_from_quaternion(current_q)
