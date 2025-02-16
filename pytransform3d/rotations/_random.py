import numpy as np

from ._axis_angle import matrix_from_compact_axis_angle
from ._matrix import check_matrix, norm_matrix
from ._utils import norm_vector


def random_vector(rng=np.random.default_rng(0), n=3):
    r"""Generate an nd vector with normally distributed components.

    Each component will be sampled from :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    n : int, optional (default: 3)
        Number of vector components

    Returns
    -------
    v : array, shape (n,)
        Random vector
    """
    return rng.standard_normal(size=n)


def random_axis_angle(rng=np.random.default_rng(0)):
    r"""Generate random axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and then the axis will be normalized
    to length 1.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : array, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    """
    angle = np.pi * rng.random()
    a = np.array([0, 0, 0, angle])
    a[:3] = norm_vector(rng.standard_normal(size=3))
    return a


def random_compact_axis_angle(rng=np.random.default_rng(0)):
    r"""Generate random compact axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and then the axis will be normalized
    to length 1.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : array, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)
    """
    a = random_axis_angle(rng)
    return a[:3] * a[3]


def random_quaternion(rng=np.random.default_rng(0)):
    """Generate random quaternion.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    return norm_vector(rng.standard_normal(size=4))


def random_matrix(rng=np.random.default_rng(0), mean=np.eye(3), cov=np.eye(3)):
    r"""Generate random rotation matrix.

    Generate :math:`\Delta \boldsymbol{R}_{B_{i+1}{B_i}}
    \boldsymbol{R}_{{B_i}A}`, with :math:`\Delta \boldsymbol{R}_{B_{i+1}{B_i}}
    = Exp(\hat{\omega} \theta)` and :math:`\hat{\omega}\theta \sim
    \mathcal{N}(\boldsymbol{0}_3, \boldsymbol{\Sigma}_{3 \times 3})`.
    The mean :math:`\boldsymbol{R}_{{B_i}A}` and the covariance
    :math:`\boldsymbol{\Sigma}_{3 \times 3}` are parameters of the function.

    Note that uncertainty is defined in the global frame B, not in the
    body frame A.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator.

    mean : array-like, shape (3, 3), optional (default: I)
        Mean rotation matrix.

    cov : array-like, shape (3, 3), optional (default: I)
        Covariance of noise in exponential coordinate space.

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    mean = check_matrix(mean)
    a = rng.multivariate_normal(mean=np.zeros(3), cov=cov)
    delta = matrix_from_compact_axis_angle(a)
    return norm_matrix(np.dot(delta, mean))
