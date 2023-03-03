import math
import numpy as np
from ..rotations import cross_product_matrix
from ._utils import check_exponential_coordinates


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


def jacobian_SE3(Stheta, check=True):
    """Jacobian of SE(3).

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    check : bool, optional (default: True)
        Check if exponential coordinates are valid
    """
    if check:
        Stheta = check_exponential_coordinates(Stheta)

    rho = Stheta[:3]
    theta = np.linalg.norm(rho)

    phi = Stheta[:3]

    Phi = cross_product_matrix(phi)
    Rho = cross_product_matrix(rho)
    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta3 * theta
    theta5 = theta4 * theta
    Q = (0.5 * cross_product_matrix(rho)
         + (theta - math.sin(theta)) / theta3 * (
                 np.dot(Phi, Rho) + np.dot(Rho, Phi)
                 + np.dot(Phi, np.dot(Rho, Phi)))
         - (1.0 - 0.5 * theta * theta - math.cos(theta)) / theta4 * (
                np.dot(Phi, np.dot(Phi, Rho))
                + np.dot(Rho, np.dot(Phi, Phi))
                - 3 * np.dot(Phi, np.dot(Rho, Phi)))
         - 0.5 * ((((1.0 - 0.5 * theta2) - math.cos(theta)) / theta4
                   - (theta - math.sin(theta) - theta3 / 6.0) / theta5)
                  * (np.dot(Phi, np.dot(Rho, np.dot(Phi, Phi)))
                     + np.dot(Phi, np.dot(Phi, np.dot(Rho, Phi)))))
         )
    J = left_jacobian_SO3(rho / theta, theta)
    return np.block([
        [J, Q],
        [np.eye(3), J]
    ])
