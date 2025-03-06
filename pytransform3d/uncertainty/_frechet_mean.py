import numpy as np

from ..batch_rotations import (
    axis_angles_from_matrices,
    matrices_from_compact_axis_angles,
)
from ..trajectories import (
    concat_many_to_one,
    exponential_coordinates_from_transforms,
    transforms_from_exponential_coordinates,
)
from ..transformations import (
    concat,
    invert_transform,
)


def frechet_mean(
    samples,
    mean0,
    exp,
    log,
    inv,
    concat_one_to_one,
    concat_many_to_one,
    n_iter=20,
):
    r"""Compute the Fréchet mean of samples on a smooth Riemannian manifold.

    The mean is computed with an iterative optimization algorithm [1]_ [2]_:

    1. For a set of samples :math:`\{x_1, \ldots, x_n\}`, we initialize
       the estimated mean :math:`\bar{x}_0`, e.g., as :math:`x_1`.
    2. For a fixed number of steps :math:`K`, in each iteration :math:`k` we
       improve the estimation of the mean by

       a. Computing the distance of each sample to the current estimate of the
          mean in tangent space with
          :math:`d_{i,k} \leftarrow \log (x_i \cdot \bar{x}_k^{-1})`.
       b. Updating the estimate of the mean with
          :math:`\bar{x}_{k+1} \leftarrow
          \exp(\frac{1}{N}\sum_i d_{i,k}) \cdot \bar{x}_k`.

    3. Return :math:`\bar{x}_K`.

    Parameters
    ----------
    samples : array-like, shape (n_samples, ...)
        Samples on a smooth Riemannian manifold.

    mean0 : array-like, shape (...)
        Initial guess for the mean on the manifold.

    exp : callable
        Exponential map from the tangent space to the manifold.

    log : callable
        Logarithmic map from the manifold to the tangent space.

    inv : callable
        Computes the inverse of an element on the manifold.

    concat_one_to_one : callable
        Concatenates elements on the manifold.

    concat_many_to_one : callable
        Concatenates multiple elements on the manifold to one element on the
        manifold.

    n_iter : int, optional (default: 20)
        Number of iterations of the optimization algorithm.

    Returns
    -------
    mean : array, shape (...)
        Fréchet mean on the manifold.

    mean_diffs : array, shape (n_samples, n_tangent_space_components)
        Differences between the mean and the samples in the tangent space.
        These can be used to compute the covariance. They are returned to
        avoid recomputing them.

    See Also
    --------
    estimate_gaussian_rotation_matrix_from_samples
        Uses the Frechet mean to compute the mean of a Gaussian distribution
        over rotations.

    estimate_gaussian_transform_from_samples
        Uses the Frechet mean to compute the mean of a Gaussian distribution
        over transformations.

    References
    ----------
    .. [1] Fréchet, M. (1948). Les éléments aléatoires de nature quelconque
       dans un espace distancié. Annales de l’Institut Henri Poincaré, 10(3),
       215–310.

    .. [2] Pennec, X. (2006). Intrinsic Statistics on Riemannian Manifolds:
       Basic Tools for Geometric Measurements. J Math Imaging Vis 25, 127-154.
       https://doi.org/10.1007/s10851-006-6228-4
    """
    assert len(samples) > 0
    samples = np.asarray(samples)
    mean = np.copy(mean0)
    for _ in range(n_iter):
        mean_diffs = log(concat_many_to_one(samples, inv(mean)))
        avg_mean_diff = np.mean(mean_diffs, axis=0)
        mean = concat_one_to_one(mean, exp(avg_mean_diff))
    return mean, mean_diffs


def estimate_gaussian_rotation_matrix_from_samples(samples):
    """Estimate Gaussian distribution over rotations from samples.

    Computes the Fréchet mean of the samples and the covariance in tangent
    space (exponential coordinates of rotation / rotation vectors) using an
    unbiased estimator as outlines by Eade [1]_.

    Parameters
    ----------
    samples : array-like, shape (n_samples, 3, 3)
        Sampled rotations represented by rotation matrices.

    Returns
    -------
    mean : array, shape (3, 3)
        Mean of the Gaussian distribution as rotation matrix.

    cov : array, shape (3, 3)
        Covariance of the Gaussian distribution in exponential coordinates.

    See Also
    --------
    frechet_mean
        Algorithm used to compute the mean of the Gaussian.

    References
    ----------
    .. [1] Eade, E. (2017). Lie Groups for 2D and 3D Transformations.
       https://ethaneade.com/lie.pdf
    """

    def compact_axis_angles_from_matrices(Rs):
        A = axis_angles_from_matrices(Rs)
        return A[:, :3] * A[:, 3, np.newaxis]

    mean, mean_diffs = frechet_mean(
        samples=samples,
        mean0=samples[0],
        exp=matrices_from_compact_axis_angles,
        log=compact_axis_angles_from_matrices,
        inv=lambda R: R.T,
        concat_one_to_one=lambda R1, R2: np.dot(R2, R1),
        concat_many_to_one=concat_many_to_one,
        n_iter=20,
    )

    cov = np.cov(mean_diffs, rowvar=False, bias=True)
    return mean, cov


def estimate_gaussian_transform_from_samples(samples):
    """Estimate Gaussian distribution over transformations from samples.

    Computes the Fréchet mean of the samples and the covariance in tangent
    space (exponential coordinates of transformation) using an unbiased
    estimator as outlines by Eade [1]_.

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

    See Also
    --------
    frechet_mean
        Algorithm used to compute the mean of the Gaussian.

    References
    ----------
    .. [1] Eade, E. (2017). Lie Groups for 2D and 3D Transformations.
       https://ethaneade.com/lie.pdf
    """
    mean, mean_diffs = frechet_mean(
        samples=samples,
        mean0=samples[0],
        exp=transforms_from_exponential_coordinates,
        log=exponential_coordinates_from_transforms,
        inv=invert_transform,
        concat_one_to_one=concat,
        concat_many_to_one=concat_many_to_one,
        n_iter=20,
    )

    cov = np.cov(mean_diffs, rowvar=False, bias=True)
    return mean, cov
