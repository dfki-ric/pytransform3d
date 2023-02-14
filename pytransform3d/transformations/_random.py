"""Random transform generation."""
import warnings
import numpy as np
from ..rotations import (
    random_quaternion, random_vector, matrix_from_quaternion, norm_vector)
from ._conversions import transform_from


def random_transform(rng=np.random.default_rng(0)):
    r"""Generate random transform.

    Each component of the translation will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Random transform from frame A to frame B
    """
    q = random_quaternion(rng)
    R = matrix_from_quaternion(q)
    p = random_vector(rng, n=3)
    return transform_from(R=R, p=p)


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
    omega = norm_vector(rng.randn(3))
    v = rng.standard_normal(size=3)
    return np.hstack((omega, v))
