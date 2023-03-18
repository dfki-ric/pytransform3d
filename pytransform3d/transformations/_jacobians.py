import math
import numpy as np
from ..rotations import (
    cross_product_matrix, left_jacobian_SO3, left_jacobian_SO3_inv)
from ._utils import check_exponential_coordinates


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

    Returns
    -------
    J : array, shape (6, 6)
        Jacobian of SE(3).
    """
    if check:
        Stheta = check_exponential_coordinates(Stheta)
    phi = Stheta[:3]
    J = left_jacobian_SO3(phi)
    return np.block([
        [J, np.zeros((3, 3))],
        [_Q(Stheta), J]
    ])


def jacobian_SE3_inv(Stheta, check=True):
    """Inverse Jacobian of SE(3).

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

    Returns
    -------
    J_inv : array, shape (6, 6)
        Inverse Jacobian of SE(3).
    """
    if check:
        Stheta = check_exponential_coordinates(Stheta)
    phi = Stheta[:3]
    J_inv = left_jacobian_SO3_inv(phi)
    return np.block([
        [J_inv, np.zeros((3, 3))],
        [-np.dot(J_inv, np.dot(_Q(Stheta), J_inv)), J_inv]
    ])


def _Q(Stheta):
    rho = Stheta[3:]
    phi = Stheta[:3]
    ph = np.linalg.norm(phi)

    px = cross_product_matrix(phi)
    rx = cross_product_matrix(rho)

    ph2 = ph * ph
    ph3 = ph2 * ph
    ph4 = ph3 * ph
    ph5 = ph4 * ph

    cph = math.cos(ph)
    sph = math.sin(ph)

    t1 = 0.5 * rx
    t2 = (ph - sph) / ph3 * (np.dot(px, rx) + np.dot(rx, px)
                             + np.dot(px, np.dot(rx, px)))
    m3 = (1.0 - 0.5 * ph * ph - cph) / ph4
    t3 = -m3 * (np.dot(px, np.dot(px, rx)) + np.dot(rx, np.dot(px, px))
                - 3 * np.dot(px, np.dot(rx, px)))
    m4 = 0.5 * (m3 - 3.0 * (ph - sph - ph3 / 6.0) / ph5)
    t4 = -m4 * (np.dot(px, np.dot(rx, np.dot(px, px)))
                + np.dot(px, np.dot(px, np.dot(rx, px))))

    Q = t1 + t2 + t3 + t4

    return Q
