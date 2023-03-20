"""Random transform generation."""
import numpy as np
from ..rotations import norm_vector
from ._utils import check_transform
from ._conversions import transform_from_exponential_coordinates


def random_transform(
        rng=np.random.default_rng(0), mean=np.eye(4), cov=np.eye(6)):
    r"""Generate random transform.

    Generate :math:`\Delta \boldsymbol{T}_{B_{i+1}{B_i}}
    \boldsymbol{T}_{{B_i}A}`, with :math:`\Delta \boldsymbol{T}_{B_{i+1}{B_i}}
    = Exp(S \theta)` and :math:`\mathcal{S}\theta \sim
    \mathcal{N}(\boldsymbol{0}_6, \boldsymbol{\Sigma}_{6 \times 6})`.
    The mean :math:`\boldsymbol{T}_{{B_i}A}` and the covariance
    :math:`\boldsymbol{\Sigma}_{6 \times 6}` are parameters of the function.

    Note that uncertainty is defined in the global frame B, not in the
    body frame A.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    mean : array-like, shape (4, 4), optional (default: I)
        Mean transform as homogeneous transformation matrix.

    cov : array-like, shape (6, 6), optional (default: I)
        Covariance of noise in exponential coordinate space.

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Random transform from frame A to frame B
    """
    mean = check_transform(mean)
    Stheta = random_exponential_coordinates(rng=rng, cov=cov)
    delta = transform_from_exponential_coordinates(Stheta)
    return np.dot(delta, mean)


def random_screw_axis(rng=np.random.default_rng(0)):
    r"""Generate random screw axis.

    Each component of v will be sampled from a standard normal distribution
    :math:`\mathcal{N}(\mu=0, \sigma=1)`. Components of :math:`\omega` will
    be sampled from a standard normal distribution and normalized.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    screw_axis : array, shape (6,)
        Screw axis described by 6 values
        (omega_1, omega_2, omega_3, v_1, v_2, v_3),
        where the first 3 components are related to rotation and the last 3
        components are related to translation.
    """
    omega = norm_vector(rng.standard_normal(size=3))
    v = rng.standard_normal(size=3)
    return np.hstack((omega, v))


def random_exponential_coordinates(
        rng=np.random.default_rng(0), cov=np.eye(6)):
    r"""Generate random exponential coordinates.

    Each component of Stheta will be sampled from a standard normal
    distribution :math:`\mathcal{N}(\boldsymbol{0}_6,
    \boldsymbol{\Sigma}_{6 \times 6})`.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    cov : array-like, shape (6, 6), optional (default: I)
        Covariance of normal distribution.

    Returns
    -------
    Stheta : array, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation. Theta should be >= 0. Negative rotations
        will be represented by a negative screw axis instead. This is relevant
        if you want to recover theta from exponential coordinates.
    """
    return rng.multivariate_normal(mean=np.zeros(6), cov=cov)
