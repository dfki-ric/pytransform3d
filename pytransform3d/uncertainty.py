import math
import numpy as np
import scipy as sp
from .transformations._lie import left_jacobian_SO3
from .transformations import (
    exponential_coordinates_from_transform,
    transform_from_exponential_coordinates,
    invert_transform, check_exponential_coordinates)
from .rotations import cross_product_matrix


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
            J_inv = np.linalg.inv(jacobian_SE3(x_ik))
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
