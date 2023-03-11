import math
import numpy as np
import scipy as sp
from .transformations import (invert_transform, check_exponential_coordinates,
                              adjoint_from_transform)
from .rotations import (cross_product_matrix, compact_axis_angle_from_matrix,
                        matrix_from_compact_axis_angle, left_jacobian_SO3,
                        left_jacobian_SO3_inv)


def fuse_poses(means, covs):
    """Fuse Gaussian distributions of poses.

    Parameters
    ----------
    means : array-like, shape (n_poses, 4, 4)
        Homogeneous transformation matrices.

    covs : array-like, shape (n_poses, 6, 6)
        Covariances of pose distributions.

    Returns
    -------
    mean : array, shape (4, 4)
        Fused pose mean.

    cov : array, shape (6, 6)
        Fused pose covariance.

    V : float
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
            x_ik = vector_from_transform(np.dot(mean, invert_transform(means[k])))
            J_inv = jacobian_SE3_inv(x_ik)
            J_invT_S = np.dot(J_inv.T, covs_inv[k])
            LHS += np.dot(J_invT_S, J_inv)
            RHS += np.dot(J_invT_S, x_ik)
        x_i = np.linalg.solve(-LHS, RHS)
        mean = np.dot(transform_from_vector(x_i), mean)

    cov = np.linalg.inv(LHS)

    V = 0.0
    for k in range(n_poses):
        x_ik = vector_from_transform(np.dot(mean, invert_transform(means[k])))
        V += 0.5 * np.dot(x_ik, np.dot(covs_inv[k], x_ik))
    return mean, cov, V


def compund_poses(T1, cov1, T2, cov2):
    """Compound two poses.

    Parameters
    ----------
    T1 : array, shape (4, 4)
        Mean of first pose.

    cov1 : array, shape (6, 6)
        Covariance of first pose.

    T2 : array, shape (4, 4)
        Mean of second pose.

    cov2 : array, shape (6, 6)
        Covariance of second pose.

    Returns
    -------
    T : array, shape (4, 4)
        Mean of new pose (T1 T2).

    cov : array, shape (6, 6)
        Covariance of new pose.
    """
    T = np.dot(T1, T2)

    ad1 = _swap_cov(adjoint_from_transform(T1))
    cov1 = _swap_cov(cov1)
    cov2 = _swap_cov(cov2)
    cov2_prime = np.dot(ad1, np.dot(cov2, ad1))

    return T, _swap_cov(cov1 + cov2_prime)

    cov1_11 = cov1[:3, :3]
    cov1_22 = cov1[3:, 3:]
    cov1_12 = cov1[:3, 3:]

    cov2_11 = cov2_prime[:3, :3]
    cov2_22 = cov2_prime[3:, 3:]
    cov2_12 = cov2_prime[:3, 3:]

    A1 = np.block([
        [_covop1(cov1_22), _covop1(np.dot(cov1_12, cov1_12.T))],
        [np.zeros((3, 3)), _covop1(cov1_22)]
    ])
    A2 = np.block([
        [_covop1(cov2_22), _covop1(np.dot(cov2_12, cov2_12.T))],
        [np.zeros((3, 3)), _covop1(cov2_22)]
    ])
    B_11 = (_covop2(cov1_22, cov2_11) + _covop2(cov1_12.T, cov2_12)
            + _covop2(cov1_12, cov2_12.T) + _covop2(cov1_11, cov2_22))
    B_12 = _covop2(cov1_22, cov2_12.T) + _covop2(cov1_12.T, cov2_22)
    B_22 = _covop2(cov1_22, cov2_22)
    B = np.block([
        [B_11, B_12],
        [B_12.T, B_22]
    ])

    cov = (
        # 2nd order
        cov1 + cov2_prime
        # 4th order
        + (np.dot(A1, cov2_prime) + np.dot(cov2_prime, A1.T)
           + np.dot(A2, cov1) + np.dot(cov1, A2.T)) / 12.0
        + B / 4.0
    )

    return T, _swap_cov(cov)


def _swap_cov(cov):
    return np.block([
        [cov[3:, 3:], cov[3:, :3]],
        [cov[:3, 3:], cov[:3, :3]]
    ])


def _covop1(A):
    return -np.trace(A) * np.eye(len(A)) + A


def _covop2(A, B):
    return np.dot(_covop1(A), _covop1(B)) + _covop1(np.dot(B, A))


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


def vector_from_transform(T):
    C = T[:3, :3]
    r = T[:3, 3]
    phi = compact_axis_angle_from_matrix(C)
    J_inv = left_jacobian_SO3_inv(phi)
    rho = np.dot(J_inv, r)
    return np.hstack((phi, rho))


def transform_from_vector(p):
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
