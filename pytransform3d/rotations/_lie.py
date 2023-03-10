import math
import numpy as np
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
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        + (1.0 - math.cos(theta)) / theta * omega_matrix
        + (1.0 - math.sin(theta) / theta) * np.dot(omega_matrix, omega_matrix)
    )


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
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        - 0.5 * omega_matrix * theta
        + (1.0 - 0.5 * theta / np.tan(theta / 2.0)) * np.dot(
            omega_matrix, omega_matrix)
    )
