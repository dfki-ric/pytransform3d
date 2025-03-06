import numpy as np

from ..transformations import (
    adjoint_from_transform,
    concat,
    invert_transform,
)


def concat_globally_uncertain_transforms(mean_A2B, cov_A2B, mean_B2C, cov_B2C):
    r"""Concatenate two independent globally uncertain transformations.

    We assume that the two distributions are independent.

    Each of the two transformations is globally uncertain (not in the local /
    body frame), that is, samples are generated through

    .. math::

        \boldsymbol{T} = Exp(\boldsymbol{\xi}) \overline{\boldsymbol{T}},

    where :math:`\boldsymbol{T} \in SE(3)` is a sampled transformation matrix,
    :math:`\overline{\boldsymbol{T}} \in SE(3)` is the mean transformation,
    and :math:`\boldsymbol{\xi} \in \mathbb{R}^6` are exponential coordinates
    of transformations and are distributed according to a Gaussian
    distribution with zero mean and covariance :math:`\boldsymbol{\Sigma} \in
    \mathbb{R}^{6 \times 6}`, that is, :math:`\boldsymbol{\xi} \sim
    \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})`.

    The concatenation order is the same as in
    :func:`~pytransform3d.transformations.concat`, that is, the transformation
    B2C is left-multiplied to A2B. Note that the order of arguments is
    different from
    :func:`~pytransform3d.uncertainty.concat_locally_uncertain_transforms`.

    Hence, the full model is

    .. math::

        Exp(_C\boldsymbol{\xi'}) \overline{\boldsymbol{T}}_{CA} =
        Exp(_C\boldsymbol{\xi}) \overline{\boldsymbol{T}}_{CB}
        Exp(_B\boldsymbol{\xi}) \overline{\boldsymbol{T}}_{BA},

    where :math:`_B\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0},
    \boldsymbol{\Sigma}_{BA})`, :math:`_C\boldsymbol{\xi} \sim
    \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_{CB})`, and
    :math:`_C\boldsymbol{\xi'} \sim \mathcal{N}(\boldsymbol{0},
    \boldsymbol{\Sigma}_{CA})`.

    This version of Barfoot and Furgale [1]_ approximates the covariance up to
    4th-order terms. Note that it is still an approximation of the covariance
    after concatenation of the two transforms.

    Parameters
    ----------
    mean_A2B : array, shape (4, 4)
        Mean of transform from A to B.

    cov_A2B : array, shape (6, 6)
        Covariance of transform from A to B. Models uncertainty in frame B.

    mean_B2C : array, shape (4, 4)
        Mean of transform from B to C.

    cov_B2C : array, shape (6, 6)
        Covariance of transform from B to C. Models uncertainty in frame C.

    Returns
    -------
    mean_A2C : array, shape (4, 4)
        Mean of new pose.

    cov_A2C : array, shape (6, 6)
        Covariance of new pose. Models uncertainty in frame C.

    See Also
    --------
    concat_locally_uncertain_transforms :
        Concatenate two independent locally uncertain transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.

    References
    ----------
    .. [1] Barfoot, T. D., Furgale, P. T. (2014).
       Associating Uncertainty With Three-Dimensional Poses for Use in
       Estimation Problems. IEEE Transactions on Robotics, 30(3), pp. 679-693,
       doi: 10.1109/TRO.2014.2298059.
    """
    mean_A2C = concat(mean_A2B, mean_B2C)

    ad_B2C = adjoint_from_transform(mean_B2C)
    cov_A2B_in_C = np.dot(ad_B2C, np.dot(cov_A2B, ad_B2C.T))
    second_order_terms = cov_B2C + cov_A2B_in_C

    cov_A2C = second_order_terms + _compound_cov_fourth_order_terms(
        cov_B2C, cov_A2B_in_C
    )

    return mean_A2C, cov_A2C


def _compound_cov_fourth_order_terms(cov1, cov2_prime):
    cov1 = _swap_cov(cov1)
    cov2_prime = _swap_cov(cov2_prime)

    cov1_11 = cov1[:3, :3]
    cov1_22 = cov1[3:, 3:]
    cov1_12 = cov1[:3, 3:]

    cov2_11 = cov2_prime[:3, :3]
    cov2_22 = cov2_prime[3:, 3:]
    cov2_12 = cov2_prime[:3, 3:]

    A1 = np.block(
        [
            [_covop1(cov1_22), _covop1(cov1_12 + cov1_12.T)],
            [np.zeros((3, 3)), _covop1(cov1_22)],
        ]
    )
    A2 = np.block(
        [
            [_covop1(cov2_22), _covop1(cov2_12 + cov2_12.T)],
            [np.zeros((3, 3)), _covop1(cov2_22)],
        ]
    )
    B_11 = (
        _covop2(cov1_22, cov2_11)
        + _covop2(cov1_12.T, cov2_12)
        + _covop2(cov1_12, cov2_12.T)
        + _covop2(cov1_11, cov2_22)
    )
    B_12 = _covop2(cov1_22, cov2_12.T) + _covop2(cov1_12.T, cov2_22)
    B_22 = _covop2(cov1_22, cov2_22)
    B = np.block([[B_11, B_12], [B_12.T, B_22]])

    return _swap_cov(
        (
            np.dot(A1, cov2_prime)
            + np.dot(cov2_prime, A1.T)
            + np.dot(A2, cov1)
            + np.dot(cov1, A2.T)
        )
        / 12.0
        + B / 4.0
    )


def _swap_cov(cov):
    return np.block([[cov[3:, 3:], cov[3:, :3]], [cov[:3, 3:], cov[:3, :3]]])


def _covop1(A):
    return -np.trace(A) * np.eye(len(A)) + A


def _covop2(A, B):
    return np.dot(_covop1(A), _covop1(B)) + _covop1(np.dot(B, A))


def concat_locally_uncertain_transforms(mean_A2B, mean_B2C, cov_A, cov_B):
    r"""Concatenate two independent locally uncertain transformations.

    We assume that the two distributions are independent.

    Each of the two transformations is locally uncertain (not in the global /
    world frame), that is, samples are generated through

    .. math::

        \boldsymbol{T} = \overline{\boldsymbol{T}} Exp(\boldsymbol{\xi}),

    where :math:`\boldsymbol{T} \in SE(3)` is a sampled transformation matrix,
    :math:`\overline{\boldsymbol{T}} \in SE(3)` is the mean transformation,
    and :math:`\boldsymbol{\xi} \in \mathbb{R}^6` are exponential coordinates
    of transformations and are distributed according to a Gaussian
    distribution with zero mean and covariance :math:`\boldsymbol{\Sigma} \in
    \mathbb{R}^{6 \times 6}`, that is, :math:`\boldsymbol{\xi} \sim
    \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma})`.

    The concatenation order is the same as in
    :func:`~pytransform3d.transformations.concat`, that is, the transformation
    B2C is left-multiplied to A2B. Note that the order of arguments is
    different from
    :func:`~pytransform3d.uncertainty.concat_globally_uncertain_transforms`.

    Hence, the full model is

    .. math::

        \overline{\boldsymbol{T}}_{CA} Exp(_A\boldsymbol{\xi'}) =
        \overline{\boldsymbol{T}}_{CB} Exp(_B\boldsymbol{\xi})
        \overline{\boldsymbol{T}}_{BA} Exp(_A\boldsymbol{\xi}),

    where :math:`_B\boldsymbol{\xi} \sim \mathcal{N}(\boldsymbol{0},
    \boldsymbol{\Sigma}_B)`, :math:`_A\boldsymbol{\xi} \sim
    \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}_A)`, and
    :math:`_A\boldsymbol{\xi'} \sim \mathcal{N}(\boldsymbol{0},
    \boldsymbol{\Sigma}_{A,total})`.

    This version of Meyer et al. [1]_ approximates the covariance up to
    2nd-order terms.

    Parameters
    ----------
    mean_A2B : array, shape (4, 4)
        Mean of transform from A to B: :math:`\overline{\boldsymbol{T}}_{BA}`.

    mean_B2C : array, shape (4, 4)
        Mean of transform from B to C: :math:`\overline{\boldsymbol{T}}_{CB}`.

    cov_A : array, shape (6, 6)
        Covariance of noise in frame A: :math:`\boldsymbol{\Sigma}_A`. Noise
        samples are right-multiplied with the mean transform A2B.

    cov_B : array, shape (6, 6)
        Covariance of noise in frame B: :math:`\boldsymbol{\Sigma}_B`. Noise
        samples are right-multiplied with the mean transform B2C.

    Returns
    -------
    mean_A2C : array, shape (4, 4)
        Mean of new pose.

    cov_A_total : array, shape (6, 6)
        Covariance of accumulated noise in frame A.

    See Also
    --------
    concat_globally_uncertain_transforms :
        Concatenate two independent globally uncertain transformations.

    pytransform3d.transformations.concat :
        Concatenate two transformations.

    References
    ----------
    .. [1] Meyer, L., Strobl, K. H., Triebel, R. (2022). The Probabilistic
       Robot Kinematics Model and its Application to Sensor Fusion.
       In IEEE/RSJ International Conference on Intelligent Robots and Systems
       (IROS), Kyoto, Japan (pp. 3263-3270),
       doi: 10.1109/IROS47612.2022.9981399.
       https://elib.dlr.de/191928/1/202212_ELIB_PAPER_VERSION_with_copyright.pdf
    """
    mean_A2C = concat(mean_A2B, mean_B2C)

    ad_B2A = adjoint_from_transform(invert_transform(mean_A2B))
    cov_B_in_A = np.dot(ad_B2A, np.dot(cov_B, ad_B2A.T))
    cov_A_total = cov_B_in_A + cov_A

    return mean_A2C, cov_A_total
