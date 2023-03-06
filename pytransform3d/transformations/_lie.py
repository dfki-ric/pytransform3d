import math
import numpy as np
from ..rotations import cross_product_matrix


def left_jacobian_SO3(omega_unit, theta):
    """Left Jacobian of SO(3).

    Parameters
    ----------
    omega_unit : array, shape (3,)
        Axis of rotation.

    theta : float
        Angle of rotation.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).
    """
    omega_matrix = cross_product_matrix(omega_unit)
    return (np.eye(3) * theta
            + (theta - math.sin(theta)) * np.dot(omega_matrix, omega_matrix)
            + (1.0 - math.cos(theta)) * omega_matrix)


def left_jacobian_SO3_inv(omega_unit, theta):
    """Inverse left Jacobian of SO(3).

    Parameters
    ----------
    omega_unit : array, shape (3,)
        Axis of rotation.

    theta : float
        Angle of rotation.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).
    """
    omega_matrix = cross_product_matrix(omega_unit)
    return (np.eye(3) / theta - 0.5 * omega_matrix
            + (1.0 / theta - 0.5 / np.tan(theta / 2.0))
            * np.dot(omega_matrix, omega_matrix))
