import math
import numpy as np
from scipy.special import bernoulli
from ._conversions import cross_product_matrix


def left_jacobian_SO3(omega):
    """Left Jacobian of SO(3) at theta (angle of rotation).

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).
    """
    omega = np.asarray(omega)
    theta = np.linalg.norm(omega)
    if theta < np.finfo(float).eps:
        return left_jacobian_SO3_series(omega, 10)
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        + (1.0 - math.cos(theta)) / theta * omega_matrix
        + (1.0 - math.sin(theta) / theta) * np.dot(omega_matrix, omega_matrix)
    )


def left_jacobian_SO3_series(omega, n_terms):
    """Left Jacobian of SO(3) at theta from Taylor series.

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    n_terms : int
        Number of terms to include in the series.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).
    """
    omega = np.asarray(omega)
    J = np.eye(3)
    pxn = np.eye(3)
    px = cross_product_matrix(omega)
    for n in range(n_terms):
        pxn = np.dot(pxn, px) / (n + 2)
        J += pxn
    return J


def left_jacobian_SO3_inv(omega):
    """Inverse left Jacobian of SO(3) at theta (angle of rotation).

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).
    """
    omega = np.asarray(omega)
    theta = np.linalg.norm(omega)
    if theta < np.finfo(float).eps:
        return left_jacobian_SO3_inv_series(omega, 10)
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        - 0.5 * omega_matrix * theta
        + (1.0 - 0.5 * theta / np.tan(theta / 2.0)) * np.dot(
            omega_matrix, omega_matrix)
    )


def left_jacobian_SO3_inv_series(omega, n_terms):
    """Inverse left Jacobian of SO(3) at theta from Taylor series.

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    n_terms : int
        Number of terms to include in the series.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).
    """
    omega = np.asarray(omega)
    J_inv = np.eye(3)
    pxn = np.eye(3)
    px = cross_product_matrix(omega)
    b = bernoulli(n_terms + 1)
    for n in range(n_terms):
        pxn = np.dot(pxn, px / (n + 1))
        J_inv += b[n + 1] * pxn
    return J_inv
