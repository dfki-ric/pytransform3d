import numpy as np

from ..transformations import (
    adjoint_from_transform,
    invert_transform,
)


def invert_uncertain_transform(mean, cov):
    r"""Invert uncertain transform.

    For the mean :math:`\boldsymbol{T}_{BA}`, the inverse is simply
    :math:`\boldsymbol{T}_{BA}^{-1} = \boldsymbol{T}_{AB}`.

    For the covariance, we need the adjoint of the inverse transformation
    :math:`\left[Ad_{\boldsymbol{T}_{BA}^{-1}}\right]`:

    .. math::

        \boldsymbol{\Sigma}_{\boldsymbol{T}_{AB}}
        =
        \left[Ad_{\boldsymbol{T}_{BA}^{-1}}\right]
        \boldsymbol{\Sigma}_{\boldsymbol{T}_{BA}}
        \left[Ad_{\boldsymbol{T}_{BA}^{-1}}\right]^T

    Parameters
    ----------
    mean : array-like, shape (4, 4)
        Mean of transform from frame A to frame B.

    cov : array, shape (6, 6)
        Covariance of transform from frame A to frame B in exponential
        coordinate space.

    Returns
    -------
    mean_inv : array, shape (4, 4)
        Mean of transform from frame B to frame A.

    cov_inv : array, shape (6, 6)
        Covariance of transform from frame B to frame A in exponential
        coordinate space.

    See Also
    --------
    pytransform3d.transformations.invert_transform :
        Invert transformation without uncertainty.

    References
    ----------
    .. [1] Mangelson, G., Vasudevan, E. (2019). Characterizing the Uncertainty
       of Jointly Distributed Poses in the Lie Algebra,
       https://arxiv.org/pdf/1906.07795.pdf
    """
    mean_inv = invert_transform(mean)
    ad_inv = adjoint_from_transform(mean_inv)
    cov_inv = np.dot(ad_inv, np.dot(cov, ad_inv.T))
    return mean_inv, cov_inv
