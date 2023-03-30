"""Operations related to uncertain transformations."""
import numpy as np
from .transformations import (
    invert_transform, transform_from, concat, adjoint_from_transform,
    left_jacobian_SE3_inv, transform_from_exponential_coordinates,
    exponential_coordinates_from_transform)
from .trajectories import (exponential_coordinates_from_transforms,
                           transforms_from_exponential_coordinates,
                           concat_many_to_one)


def estimate_gaussian_transform_from_samples(samples):
    """Estimate Gaussian distribution over transformations from samples.

    Uses iterative approximation of mean described by Eade (2017) and computes
    covariance in exponential coordinate space (using an unbiased estimator).

    Parameters
    ----------
    samples : array-like, shape (n_samples, 4, 4)
        Sampled transformations represented by homogeneous matrices.

    Returns
    -------
    mean : array, shape (4, 4)
        Mean as homogeneous transformation matrix.

    cov : array, shape (6, 6)
        Covariance of distribution in exponential coordinate space.

    References
    ----------
    Eade: Lie Groups for 2D and 3D Transformations (2017),
    https://ethaneade.com/lie.pdf
    """
    assert len(samples) > 0
    mean = samples[0]
    for _ in range(20):
        mean_inv = invert_transform(mean)
        mean_diffs = exponential_coordinates_from_transforms(
            concat_many_to_one(samples, mean_inv))
        avg_mean_diff = np.mean(mean_diffs, axis=0)
        mean = np.dot(
            transform_from_exponential_coordinates(avg_mean_diff), mean)

    cov = np.cov(mean_diffs, rowvar=False, bias=True)
    return mean, cov


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


def concat_uncertain_transforms(mean_A2B, cov_A2B, mean_B2C, cov_B2C):
    """Concatenate two independent uncertain transformations.

    This version of Barfoot and Furgale approximates the covariance up to
    4th-order terms. Note that it is still an approximation of the covariance
    after concatenation of the two transforms.

    We assume that the two distributions are independent.

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
        [_covop1(cov1_22), _covop1(cov1_12 + cov1_12.T)],
        [np.zeros((3, 3)), _covop1(cov1_22)]
    ])
    A2 = np.block([
        [_covop1(cov2_22), _covop1(cov2_12 + cov2_12.T)],
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
    LHS = np.empty((6, 6))
    RHS = np.empty(6)
    for _ in range(20):
        LHS[:, :] = 0.0
        RHS[:] = 0.0
        for k in range(n_poses):
            x_ik = exponential_coordinates_from_transform(
                np.dot(mean, invert_transform(means[k])))
            J_inv = left_jacobian_SE3_inv(x_ik)
            J_invT_S = np.dot(J_inv.T, covs_inv[k])
            LHS += np.dot(J_invT_S, J_inv)
            RHS += np.dot(J_invT_S, x_ik)
        x_i = np.linalg.solve(-LHS, RHS)
        mean = np.dot(transform_from_exponential_coordinates(x_i), mean)

    cov = np.linalg.inv(LHS)

    V = 0.0
    for k in range(n_poses):
        x_ik = exponential_coordinates_from_transform(
            np.dot(mean, invert_transform(means[k])))
        V += 0.5 * np.dot(x_ik, np.dot(covs_inv[k], x_ik))
    return mean, cov, V


def to_ellipsoid(mean, cov):
    """Compute error ellipsoid.

    An error ellipsoid indicates the equiprobable surface. The resulting
    ellipsoid includes one standard deviation of the data along each main
    axis, which covers approximately 68.27% of the data. Multiplying the
    radii with factors > 1 will increase the coverage. The usual factors
    for Gaussian distributions apply:

    * 1 - 68.27%
    * 1.65 - 90%
    * 1.96 - 95%
    * 2 - 95.45%
    * 2.58 - 99%
    * 3 - 99.73%

    Parameters
    ----------
    mean : array-like, shape (3,)
        Mean of distribution.

    cov : array-like, shape (3, 3)
        Covariance of distribution.

    Returns
    -------
    ellipsoid2origin : array, shape (4, 4)
        Ellipsoid frame in world frame. Note that there are multiple solutions
        possible for the orientation because an ellipsoid is symmetric.
        A body-fixed rotation around a main axis by 180 degree results in the
        same ellipsoid.

    radii : array, shape (3,)
        Radii of ellipsoid, coinciding with standard deviations along the
        three axes of the ellipsoid. These are sorted in ascending order.
    """
    from scipy import linalg
    radii, R = linalg.eigh(cov)
    if np.linalg.det(R) < 0:  # undo reflection (exploit symmetry)
        R *= -1
    ellipsoid2origin = transform_from(R=R, p=mean)
    return ellipsoid2origin, np.sqrt(np.abs(radii))


def to_projected_ellipsoid(mean, cov, factor=1.96, n_steps=20):
    """Compute projected error ellipsoid.

    An error ellipsoid shows equiprobable points. This is a projection of a
    Gaussian distribution in exponential coordinate space to 3D.

    Parameters
    ----------
    mean : array-like, shape (4, 4)
        Mean of pose distribution.

    cov : array-like, shape (6, 6)
        Covariance of pose distribution in exponential coordinate space.

    factor : float, optional (default: 1.96)
        Multiple of the standard deviations that should be plotted.

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension.

    Returns
    -------
    x : array, shape (n_steps, n_steps)
        Coordinates on x-axis of grid on projected ellipsoid.

    y : array, shape (n_steps, n_steps)
        Coordinates on y-axis of grid on projected ellipsoid.

    z : array, shape (n_steps, n_steps)
        Coordinates on z-axis of grid on projected ellipsoid.
    """
    from scipy import linalg
    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    radii = factor * np.sqrt(vals)

    # Grid on ellipsoid in exponential coordinate space
    radius_x, radius_y, radius_z = radii[:3]
    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = radius_x * np.sin(phi) * np.cos(theta)
    y = radius_y * np.sin(phi) * np.sin(theta)
    z = radius_z * np.cos(phi)
    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    P = np.dot(P, vecs[:, :3].T)

    # Grid in Cartesian space
    T_diff = transforms_from_exponential_coordinates(P)
    # same as T_diff[m, :3, :3].T.dot(T_diff[m, :3, 3]) for each m
    P = np.einsum("ikj,ik->ij", T_diff[:, :3, :3], T_diff[:, :3, 3])
    P = (np.dot(P, mean[:3, :3].T) + mean[np.newaxis, :3, 3]).T

    shape = x.shape
    x = P[0].reshape(*shape)
    y = P[1].reshape(*shape)
    z = P[2].reshape(*shape)

    return x, y, z


def plot_projected_ellipsoid(
        ax, mean, cov, factor=1.96, wireframe=True, n_steps=20, color=None,
        alpha=1.0):  # pragma: no cover
    """Plots projected equiprobable ellipsoid in 3D.

    An error ellipsoid shows equiprobable points. This is a projection of a
    Gaussian distribution in exponential coordinate space to 3D.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mean : array-like, shape (4, 4)
        Mean pose.

    cov : array-like, shape (6, 6)
        Covariance in exponential coordinate space.

    factor : float, optional (default: 1.96)
        Multiple of the standard deviations that should be plotted.

    wireframe : bool, optional (default: True)
        Plot wireframe of ellipsoid and surface otherwise.

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension.

    color : str, optional (default: None)
        Color in which the equiprobably lines should be plotted.

    alpha : float, optional (default: 1.0)
        Alpha value for lines.

    Returns
    -------
    ax : axis
        Matplotlib axis.
    """
    x, y, z = to_projected_ellipsoid(mean, cov, factor, n_steps)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=2, cstride=2, color=color, alpha=alpha)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax
