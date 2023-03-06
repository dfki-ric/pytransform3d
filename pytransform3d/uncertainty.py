import math
import numpy as np
import scipy as sp
from .transformations import invert_transform, check_exponential_coordinates
from .rotations import (cross_product_matrix, compact_axis_angle_from_matrix,
                        matrix_from_compact_axis_angle, eps)


def fuse_poses(means, covs, return_error=False):
    """Fuse Gaussian distributions of poses.

    Parameters
    ----------
    means : array-like, shape (n_poses, 4, 4)
        Homogeneous transformation matrices.

    covs : array-like, shape (n_poses, 6, 6)
        Covariances of pose distributions.

    return_error : bool, optional (default: False)
        Return error of optimization objective.

    Returns
    -------
    mean : array, shape (4, 4)
        Fused pose mean.

    cov : array, shape (6, 6)
        Fused pose covariance.

    V : float, optional
        Error of optimization objective.

    References
    ----------
    Barfoot, Furgale: Associating Uncertainty With Three-Dimensional Poses for
    Use in Estimation Problems,
    http://ncfrn.mcgill.ca/members/pubs/barfoot_tro14.pdf
    """
    n_poses = len(means)
    covs_inv = [np.linalg.inv(cov) for cov in covs]

    mean = np.eye(4)
    for i in range(20):
        LHS = np.zeros((6, 6))
        RHS = np.zeros(6)
        for k in range(n_poses):
            x_ik = tran2vec(np.dot(mean, invert_transform(means[k])))
            J_inv = jacobian_SE3_inv(x_ik)
            J_invT_S = np.dot(J_inv.T, covs_inv[k])
            LHS += np.dot(J_invT_S, J_inv)
            RHS += np.dot(J_invT_S, x_ik)
        x_i = np.linalg.solve(-LHS, RHS)
        mean = np.dot(vec2tran(x_i), mean)

    cov = np.linalg.inv(LHS)
    if return_error:
        V = 0.0
        for k in range(n_poses):
            x_ik = tran2vec(np.dot(mean, invert_transform(means[k])))
            V += 0.5 * np.dot(x_ik, np.dot(covs_inv[k], x_ik))
        return mean, cov, V
    else:
        return mean, cov


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
    if angle < eps:
        return np.eye(3)  # TODO check

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
        [J, np.zeros((3, 3))],
        [_Q(Stheta), J]
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
    if angle < eps:
        return np.eye(3)  # TODO check

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


def tran2vec(T):
    C = T[:3, :3]
    r = T[:3, 3]
    phi = compact_axis_angle_from_matrix(C)
    J_inv = left_jacobian_SO3_inv(phi)
    rho = np.dot(J_inv, r)
    return np.hstack((phi, rho))


def rot2vec(C):
    d, v = np.linalg.eig(C)
    for i in range(3):
        if abs(d[i] - 1) < 1e-10:
            a = np.real(v[:, i])
            a /= np.linalg.norm(a)
            phim = np.arccos(0.5 * (np.trace(C) - 1.0))
            phi = phim * a
            if abs(np.trace(np.dot(vec2rot(phi).T, C)) - 3) > 1e-14:
                phi *= -1.0
            return phi


def vec2rot(phi):
    angle = np.linalg.norm(phi)
    axis = phi / angle
    cp = math.cos(angle)
    sp = math.sin(angle)
    C = cp * np.eye(3) + (1 - cp) * np.outer(axis, axis) + sp * cross_product_matrix(axis)
    return C


def vec2tran(p):
    rho = p[3:]
    phi = p[:3]
    C = matrix_from_compact_axis_angle(phi)
    J = left_jacobian_SO3(phi)
    T = np.eye(4)
    T[:3, :3] = C
    T[:3, 3] = np.dot(J, rho)
    return T


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
