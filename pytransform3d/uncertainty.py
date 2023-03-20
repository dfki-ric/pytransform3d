import math
import numpy as np
from .transformations import (
    invert_transform, concat, adjoint_from_transform, left_jacobian_SE3_inv,
    transform_from_exponential_coordinates,
    exponential_coordinates_from_transform)
from .trajectories import exponential_coordinates_from_transforms


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

    References
    ----------
    Mangelson, Ghaffari, Vasudevan, Eustice: Characterizing the Uncertainty of
    Jointly Distributed Poses in the Lie Algebra,
    https://arxiv.org/pdf/1906.07795.pdf
    """
    mean_inv = invert_transform(mean)
    ad_inv = adjoint_from_transform(mean_inv)
    cov_inv = np.dot(ad_inv, np.dot(cov, ad_inv.T))
    return mean_inv, cov_inv


def pose_fusion(means, covs):
    """Fuse Gaussian distributions of multiple poses.

    Parameters
    ----------
    means : array-like, shape (n_poses, 4, 4)
        Homogeneous transformation matrices.

    covs : array-like, shape (n_poses, 6, 6)
        Covariances of pose distributions in exponential coordinate space.

    Returns
    -------
    mean : array, shape (4, 4)
        Fused pose mean.

    cov : array, shape (6, 6)
        Fused pose covariance.

    V : float
        Error of optimization objective.

    References
    ----------
    Barfoot, Furgale: Associating Uncertainty With Three-Dimensional Poses for
    Use in Estimation Problems,
    http://ncfrn.mcgill.ca/members/pubs/barfoot_tro14.pdf
    """
    n_poses = len(means)
    covs_inv = [np.linalg.inv(cov) for cov in covs]

    mean = np.eye(4)
    for i in range(20):
        LHS = np.zeros((6, 6))
        RHS = np.zeros(6)
        for k in range(n_poses):
            x_ik = exponential_coordinates_from_transform(np.dot(mean, invert_transform(means[k])))
            J_inv = left_jacobian_SE3_inv(x_ik)
            J_invT_S = np.dot(J_inv.T, covs_inv[k])
            LHS += np.dot(J_invT_S, J_inv)
            RHS += np.dot(J_invT_S, x_ik)
        x_i = np.linalg.solve(-LHS, RHS)
        mean = np.dot(transform_from_exponential_coordinates(x_i), mean)

    cov = np.linalg.inv(LHS)

    V = 0.0
    for k in range(n_poses):
        x_ik = exponential_coordinates_from_transform(np.dot(mean, invert_transform(means[k])))
        V += 0.5 * np.dot(x_ik, np.dot(covs_inv[k], x_ik))
    return mean, cov, V


def concat_uncertain_transforms(mean_A2B, cov_A2B, mean_B2C, cov_B2C):
    """Concatenate two independent uncertain transformations.

    Parameters
    ----------
    mean_A2B : array, shape (4, 4)
        Mean of transform from A to B.

    cov_A2B : array, shape (6, 6)
        Covariance of transform from A to B.

    mean_B2C : array, shape (4, 4)
        Mean of transform from B to C.

    cov_B2C : array, shape (6, 6)
        Covariance of transform from B to C.

    Returns
    -------
    mean_A2C : array, shape (4, 4)
        Mean of new pose.

    cov_A2C : array, shape (6, 6)
        Covariance of new pose.

    References
    ----------
    Barfoot, Furgale: Associating Uncertainty With Three-Dimensional Poses for
    Use in Estimation Problems,
    http://ncfrn.mcgill.ca/members/pubs/barfoot_tro14.pdf
    """
    mean_A2C = concat(mean_A2B, mean_B2C)

    ad_B2C = adjoint_from_transform(mean_B2C)
    cov_A2B_in_C = np.dot(ad_B2C, np.dot(cov_A2B, ad_B2C.T))
    second_order_terms = cov_B2C + cov_A2B_in_C

    cov_A2C = second_order_terms + _compound_cov_fourth_order_terms(
        cov_B2C, cov_A2B_in_C)

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

    A1 = np.block([
        [_covop1(cov1_22), _covop1(np.dot(cov1_12, cov1_12.T))],
        [np.zeros((3, 3)), _covop1(cov1_22)]
    ])
    A2 = np.block([
        [_covop1(cov2_22), _covop1(np.dot(cov2_12, cov2_12.T))],
        [np.zeros((3, 3)), _covop1(cov2_22)]
    ])
    B_11 = (_covop2(cov1_22, cov2_11) + _covop2(cov1_12.T, cov2_12)
            + _covop2(cov1_12, cov2_12.T) + _covop2(cov1_11, cov2_22))
    B_12 = _covop2(cov1_22, cov2_12.T) + _covop2(cov1_12.T, cov2_22)
    B_22 = _covop2(cov1_22, cov2_22)
    B = np.block([
        [B_11, B_12],
        [B_12.T, B_22]
    ])

    return _swap_cov(
        (np.dot(A1, cov2_prime) + np.dot(cov2_prime, A1.T)
         + np.dot(A2, cov1) + np.dot(cov1, A2.T)) / 12.0
        + B / 4.0
    )


def _swap_cov(cov):
    return np.block([
        [cov[3:, 3:], cov[3:, :3]],
        [cov[:3, 3:], cov[:3, :3]]
    ])


def _covop1(A):
    return -np.trace(A) * np.eye(len(A)) + A


def _covop2(A, B):
    return np.dot(_covop1(A), _covop1(B)) + _covop1(np.dot(B, A))


def concat_dependent_uncertain_transforms(
        mean_A2B, cov_A2B, mean_B2C, cov_B2C, cov_A2B_B2C):
    """Concatenate two dependent uncertain transformations.

    Parameters
    ----------
    mean_A2B : array, shape (4, 4)
        Mean of transform from A to B.

    cov_A2B : array, shape (6, 6)
        Covariance of transform from A to B.

    mean_B2C : array, shape (4, 4)
        Mean of transform from B to C.

    cov_B2C : array, shape (6, 6)
        Covariance of transform from B to C.

    cov_A2B_B2C : array, shape (6, 6)
        Covariance between the two transforms.

    Returns
    -------
    mean_A2C : array, shape (4, 4)
        Mean of new pose.

    cov_A2C : array, shape (6, 6)
        Covariance of new pose.

    References
    ----------
    Mangelson, Ghaffari, Vasudevan, Eustice: Characterizing the Uncertainty of
    Jointly Distributed Poses in the Lie Algebra,
    https://arxiv.org/pdf/1906.07795.pdf
    """
    mean_A2C = concat(mean_A2B, mean_B2C)

    ad_B2C = adjoint_from_transform(mean_B2C)
    cov_A2B_in_C = np.dot(ad_B2C, np.dot(cov_A2B, ad_B2C.T))
    cov_A2C = (
        cov_B2C + cov_A2B_in_C
        + np.dot(cov_A2B_B2C, ad_B2C.T) + np.dot(ad_B2C, cov_A2B_B2C.T))

    return mean_A2C, cov_A2C


def estimate_gaussian_transform_from_samples(samples):
    """Estimate Gaussian distribution over transformations from samples.

    Parameters
    ----------
    samples : array-like, shape (n_samples, 4, 4)
        Sampled transformations represented by homogeneous matrices.

    Returns
    -------
    mean : array, shape (4, 4)
        Homogeneous transformation matrix.

    cov : array, shape (6, 6)
        Covariance of distribution in exponential coordinate space.
    """
    exp_coord_samples = exponential_coordinates_from_transforms(samples)
    mean = transform_from_exponential_coordinates(
        np.mean(exp_coord_samples, axis=0))
    cov = np.cov(exp_coord_samples, rowvar=False, bias=True)
    return mean, cov


def to_ellipse(cov, factor=1.0):
    """Compute error ellipse.

    An error ellipse shows equiprobable points of a 2D Gaussian distribution.

    Parameters
    ----------
    cov : array-like, shape (2, 2)
        Covariance of the Gaussian distribution.

    factor : float
        One means standard deviation.

    Returns
    -------
    angle : float
        Rotation angle of the ellipse.

    width : float
        Width of the ellipse (semi axis, not diameter).

    height : float
        Height of the ellipse (semi axis, not diameter).
    """
    from scipy import linalg
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.arctan2(*vecs[:, 0][::-1])
    width, height = factor * np.sqrt(vals)
    return angle, width, height


def plot_error_ellipse(ax, mean, cov, color=None, alpha=0.25,
                       factors=np.linspace(0.25, 2.0, 8)):
    """Plot error ellipse of MVN.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mean : array-like, shape (2,)
        Mean of the Gaussian distribution.

    cov : array-like, shape (2, 2)
        Covariance of the Gaussian distribution.

    color : str, optional (default: None)
        Color in which the ellipse should be plotted

    alpha : float, optional (default: 0.25)
        Alpha value for ellipse

    factors : array, optional (default: np.linspace(0.25, 2.0, 8))
        Multiples of the standard deviations that should be plotted.
    """
    from matplotlib.patches import Ellipse
    for factor in factors:
        angle, width, height = to_ellipse(cov, factor)
        ell = Ellipse(xy=mean, width=2.0 * width, height=2.0 * height,
                      angle=np.degrees(angle))
        ell.set_alpha(alpha)
        if color is not None:
            ell.set_color(color)
        ax.add_artist(ell)


def plot_projected_ellipse(
        ax, mean, cov, dimensions, color=None, alpha=0.25, factor=1.96):
    """Plots projected great circles of equiprobable ellipsoid in 2D.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mean : array-like, shape (4, 4)
        Mean pose.

    cov : array-like, shape (6, 6)
        Covariance in vector space.

    dimensions : array, (2,)
        Output dimensions.

    color : str, optional (default: None)
        Color in which the equiprobably lines should be plotted.

    alpha : float, optional (default: 0.25)
        Alpha value for lines.

    factor : float, optional (default: 1.96)
        Multiple of the standard deviations that should be plotted.
    """
    vals, vecs = np.linalg.eig(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    w = factor * np.sqrt(vals[:3])

    n_steps = 50
    ind1 = [0, 1, 0]
    ind2 = [1, 2, 2]
    V = -math.pi + 2 * math.pi * (np.arange(n_steps) - 1) / (n_steps - 1)
    S = np.sin(V)
    C = np.cos(V)
    for n in range(3):
        clines = np.zeros((2, n_steps))
        P = (w[ind1[n]] * vecs[np.newaxis, :, ind1[n]] * S[:, np.newaxis]
             + w[ind2[n]] * vecs[np.newaxis, :, ind2[n]] * C[:, np.newaxis])
        for m in range(n_steps):
            T = transform_from_exponential_coordinates(P[m]).dot(mean)
            r = T[:3, :3].T.dot(T[:3, 3])
            clines[:, m] = r[dimensions]
        ax.plot(clines[0], clines[1], color=color, alpha=alpha)
