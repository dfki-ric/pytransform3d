import numpy as np

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
