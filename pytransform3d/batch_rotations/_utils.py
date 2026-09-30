"""Utility functions."""

import numpy as np


def norm_vectors(V, out=None):
    """Normalize vectors.

    Parameters
    ----------
    V : array-like, shape (..., n)
        nd vectors

    out : array, shape (..., n), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    V_unit : array, shape (..., n)
        nd unit vectors with norm 1 or zero vectors
    """
    V = np.asarray(V)
    norms = np.linalg.norm(V, axis=-1)
    if out is None:
        # Integer inputs require floating-point output; retain floating dtypes.
        out = np.empty_like(V, dtype=np.result_type(V.dtype, norms.dtype))
    # Avoid division by zero with np.maximum(..., smallest positive float).
    # The norm is zero only when the vector is zero so this case does not
    # require further processing.
    out[...] = V / np.maximum(norms[..., np.newaxis], np.finfo(float).tiny)
    return out


def angles_between_vectors(A, B):
    r"""Compute angle between two vectors.

    .. math::

        \theta =
        2 \operatorname{atan2}
        \left(
            \left\|
                \|b\| a - \|a\| b
            \right\|,
            \left\|
                \|b\| a + \|a\| b
            \right\|
        \right)

    See [1]_, page 47. To simplify the implementation, we normalize both
    vectors first such that norms of 1 cancel.

    Parameters
    ----------
    A : array-like, shape (..., n)
        nd vectors

    B : array-like, shape (..., n)
        nd vectors

    Returns
    -------
    angles : array, shape (...)
        Angles between pairs of vectors from A and B

    References
    ----------
    .. [1] Kahan, W. (2006).
       How Futile are Mindless Assessments of Roundoff in Floating-Point
       Computation?
       https://people.eecs.berkeley.edu/~wkahan/Mindless.pdf
    """
    A = np.asarray(A)
    B = np.asarray(B)
    A = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B = B / np.linalg.norm(B, axis=-1, keepdims=True)
    return 2.0 * np.arctan2(
        np.linalg.norm(A - B, axis=-1),
        np.linalg.norm(A + B, axis=-1),
    )


def cross_product_matrices(V):
    """Generate the cross-product matrices of vectors.

    The cross-product matrix :math:`\\boldsymbol{V}` satisfies the equation

    .. math::

        \\boldsymbol{V} \\boldsymbol{w} = \\boldsymbol{v} \\times
        \\boldsymbol{w}

    It is a skew-symmetric (antisymmetric) matrix, i.e.
    :math:`-\\boldsymbol{V} = \\boldsymbol{V}^T`.

    Parameters
    ----------
    V : array-like, shape (..., 3)
        3d vectors

    Returns
    -------
    V_cross_product_matrices : array, shape (..., 3, 3)
        Cross-product matrices of V
    """
    V = np.asarray(V)

    instances_shape = V.shape[:-1]
    V_matrices = np.empty(instances_shape + (3, 3))

    V_matrices[..., 0, 0] = 0.0
    V_matrices[..., 0, 1] = -V[..., 2]
    V_matrices[..., 0, 2] = V[..., 1]
    V_matrices[..., 1, 0] = V[..., 2]
    V_matrices[..., 1, 1] = 0.0
    V_matrices[..., 1, 2] = -V[..., 0]
    V_matrices[..., 2, 0] = -V[..., 1]
    V_matrices[..., 2, 1] = V[..., 0]
    V_matrices[..., 2, 2] = 0.0

    return V_matrices
