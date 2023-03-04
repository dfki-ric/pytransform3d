import math
import numpy as np
import scipy as sp
from .transformations import (
    exponential_coordinates_from_transform,
    transform_from_exponential_coordinates,
    invert_transform, check_exponential_coordinates)
from .rotations import cross_product_matrix


def left_jacobian_SO3(omega):
    """Left Jacobian of SO(3).

    Parameters
    ----------
    omega : array, shape (3,)
        Compact axis-angle representation

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).
    """
    angle = np.linalg.norm(omega)
    axis = omega / angle

    cph = (1.0 - math.cos(angle)) / angle
    sph = math.sin(angle) / angle

    return (
        sph * np.eye(3)
        + (1.0 - sph) * np.outer(axis, axis)
        + cph * cross_product_matrix(axis)
    )


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
        [J, _Q(Stheta)],
        [np.zeros((3, 3)), J]
    ])


def left_jacobian_SO3_inv(omega):
    """Inverse left Jacobian of SO(3).

    Parameters
    ----------
    omega : array, shape (3,)
        Compact axis-angle representation

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).
    """
    angle = np.linalg.norm(omega)
    axis = omega / angle
    angle_2 = 0.5 * angle
    return (
        angle_2 / math.tan(angle_2) * np.eye(3)
        + (1.0 - angle_2 / math.tan(angle_2)) * np.outer(axis, axis)
        - angle_2 * cross_product_matrix(axis)
    )


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
        [J_inv, -np.dot(J_inv, np.dot(_Q(Stheta), J_inv))],
        [np.zeros((3, 3)), J_inv]
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


def fuse_poses(means, covs, return_error=False):
    """TODO"""
    n_poses = len(means)

    covs_inv = [np.linalg.inv(cov) for cov in covs]

    mean = np.eye(4)
    for i in range(20):
        LHS = np.zeros((6, 6))
        RHS = np.zeros(6)
        for k in range(n_poses):
            x_ik = exponential_coordinates_from_transform(
                np.dot(mean, invert_transform(means[k])))
            J_inv = jacobian_SE3_inv(x_ik)
            J_invT_S = np.dot(J_inv.T, covs_inv[k])
            LHS += np.dot(J_invT_S, J_inv)
            RHS += np.dot(J_invT_S, x_ik)
        x_i = np.linalg.solve(-LHS, RHS)
        mean = np.dot(transform_from_exponential_coordinates(x_i), mean)

    V = 0.0
    for k in range(n_poses):
        x_ik = exponential_coordinates_from_transform(
            np.dot(mean, invert_transform(means[k])))
        V += 0.5 * np.dot(x_ik, np.dot(covs_inv[k], x_ik))

    cov = np.linalg.inv(LHS)
    if return_error:
        return mean, cov, V
    else:
        return mean, cov


def to_ellipse(cov, factor=1.0):
    """Compute error ellipse.

    An error ellipse shows equiprobable points of a 2D Gaussian distribution.

    Parameters
    ----------
    cov : array-like, shape (2, 2)
        Covariance of the Gaussian distribution.

    factor : float
        One means standard deviation.

    Returns
    -------
    angle : float
        Rotation angle of the ellipse.

    width : float
        Width of the ellipse (semi axis, not diameter).

    height : float
        Height of the ellipse (semi axis, not diameter).
    """
    vals, vecs = sp.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.arctan2(*vecs[:, 0][::-1])
    width, height = factor * np.sqrt(vals)
    return angle, width, height


def plot_error_ellipse(ax, mean, cov, color=None, alpha=0.25,
                       factors=np.linspace(0.25, 2.0, 8)):
    """Plot error ellipse of MVN.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mean : array-like, shape (2,)
        Mean of the Gaussian distribution.

    cov : array-like, shape (2, 2)
        Covariance of the Gaussian distribution.

    color : str, optional (default: None)
        Color in which the ellipse should be plotted

    alpha : float, optional (default: 0.25)
        Alpha value for ellipse

    factors : array, optional (default: np.linspace(0.25, 2.0, 8))
        Multiples of the standard deviations that should be plotted.
    """
    from matplotlib.patches import Ellipse
    for factor in factors:
        angle, width, height = to_ellipse(cov, factor)
        ell = Ellipse(xy=mean, width=2.0 * width, height=2.0 * height,
                      angle=np.degrees(angle))
        ell.set_alpha(alpha)
        if color is not None:
            ell.set_color(color)
        ax.add_artist(ell)
