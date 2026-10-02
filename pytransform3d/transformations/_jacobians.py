"""Jacobians of SE(3)."""

import math

import numpy as np

from ._screws import check_exponential_coordinates
from ..rotations import (
    cross_product_matrix,
    left_jacobian_SO3,
    left_jacobian_SO3_inv,
)


def left_jacobian_SE3(Stheta):
    r"""Left Jacobian of SE(3).

    .. math::

        \boldsymbol{\mathcal{J}}
        =
        \left(
        \begin{array}{cc}
        \boldsymbol{J} & \boldsymbol{0}\\
        \boldsymbol{Q} & \boldsymbol{J}
        \end{array}
        \right),

    where :math:`\boldsymbol{J}` is the left Jacobian of SO(3) and
    :math:`\boldsymbol{Q}` is given by Barfoot and Furgale (see reference
    below).

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    Returns
    -------
    J : array, shape (6, 6)
        Jacobian of SE(3).

    See Also
    --------
    left_jacobian_SE3_series :
        Left Jacobian of SE(3) at theta from Taylor series.

    left_jacobian_SE3_inv : Left inverse Jacobian of SE(3).

    References
    ----------
    .. [1] Barfoot, T. D., Furgale, P. T. (2014).
       Associating Uncertainty With Three-Dimensional Poses for Use in
       Estimation Problems. IEEE Transactions on Robotics, 30(3), pp. 679-693,
       doi: 10.1109/TRO.2014.2298059.
    """
    Stheta = check_exponential_coordinates(Stheta)

    # The series depends on the rotation angle, not the screw parameter,
    # which is the translation magnitude for pure translations.
    theta = np.linalg.norm(Stheta[:3])
    if theta < math.sqrt(6.0 * np.finfo(float).eps):
        return left_jacobian_SE3_series(Stheta, 10)

    phi = Stheta[:3]
    J = left_jacobian_SO3(phi)
    return np.block([[J, np.zeros((3, 3))], [_Q(Stheta), J]])


def left_jacobian_SE3_series(Stheta, n_terms):
    """Left Jacobian of SE(3) at theta from Taylor series.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    n_terms : int
        Number of terms to include in the series.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SE(3).

    See Also
    --------
    left_jacobian_SE3 : Left Jacobian of SE(3).
    """
    Stheta = check_exponential_coordinates(Stheta)
    J = np.eye(6)
    pxn = np.eye(6)
    px = _curlyhat(Stheta)
    for n in range(n_terms):
        pxn = np.dot(pxn, px) / (n + 2)
        J += pxn
    return J


def left_jacobian_SE3_inv(Stheta):
    r"""Left inverse Jacobian of SE(3).

    .. math::

        \boldsymbol{\mathcal{J}}^{-1}
        =
        \left(
        \begin{array}{cc}
        \boldsymbol{J}^{-1} & \boldsymbol{0}\\
        -\boldsymbol{J}^{-1}\boldsymbol{Q}\boldsymbol{J}^{-1} &
        \boldsymbol{J}^{-1}
        \end{array}
        \right),

    where :math:`\boldsymbol{J}` is the left Jacobian of SO(3) and
    :math:`\boldsymbol{Q}` is given by Barfoot and Furgale (see reference
    below).

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    Returns
    -------
    J_inv : array, shape (6, 6)
        Inverse Jacobian of SE(3).

    See Also
    --------
    left_jacobian_SE3 : Left Jacobian of SE(3).

    left_jacobian_SE3_inv_series :
        Left inverse Jacobian of SE(3) at theta from Taylor series.
    """
    Stheta = check_exponential_coordinates(Stheta)

    theta = np.linalg.norm(Stheta[:3])
    if theta < math.sqrt(6.0 * np.finfo(float).eps):
        return left_jacobian_SE3_inv_series(Stheta, 10)

    phi = Stheta[:3]
    J_inv = left_jacobian_SO3_inv(phi)
    return np.block(
        [
            [J_inv, np.zeros((3, 3))],
            [-np.dot(J_inv, np.dot(_Q(Stheta), J_inv)), J_inv],
        ]
    )


def _Q(Stheta):
    rho = Stheta[3:]
    phi = Stheta[:3]
    ph = np.linalg.norm(phi)

    px = cross_product_matrix(phi)
    rx = cross_product_matrix(rho)

    ph2 = ph * ph
    if ph < 0.5:
        # ph is the rotation angle in radians. Expand the closed forms
        # below about ph = 0:
        # m2 = (ph - sin(ph))/ph**3 = 1/6 - ph**2/120 + ...
        # m3 = (1 - ph**2/2 - cos(ph))/ph**4 = -1/24 + ph**2/720 + ...
        # m4 = (m3 - 3*(ph - sin(ph) - ph**3/6)/ph**5)/2
        #    = -1/120 + ph**2/2520 + ...
        # Horner evaluation through ph**10 avoids cancellation in the
        # small sin/cos remainders before division by powers of ph.
        # On 0 <= ph <= 0.5, these alternating series decrease in magnitude,
        # so the first omitted terms bound the absolute truncation errors:
        # ph**12/15!, ph**12/16!, and 7*ph**12/17!, respectively. At 0.5,
        # they are 1.87e-16, 1.17e-17, and 4.80e-18, all below float64
        # machine epsilon (2.22e-16). This motivates the 0.5 cutoff.
        m2 = 1.0 / 6.0 - ph2 * (
            1.0 / 120.0
            - ph2
            * (
                1.0 / 5040.0
                - ph2
                * (
                    1.0 / 362880.0
                    - ph2 * (1.0 / 39916800.0 - ph2 / 6227020800.0)
                )
            )
        )
        m3 = -1.0 / 24.0 + ph2 * (
            1.0 / 720.0
            - ph2
            * (
                1.0 / 40320.0
                - ph2
                * (
                    1.0 / 3628800.0
                    - ph2 * (1.0 / 479001600.0 - ph2 / 87178291200.0)
                )
            )
        )
        m4 = -1.0 / 120.0 + ph2 * (
            1.0 / 2520.0
            - ph2
            * (
                1.0 / 120960.0
                - ph2
                * (
                    1.0 / 9979200.0
                    - ph2 * (1.0 / 1245404160.0 - ph2 / 217945728000.0)
                )
            )
        )
    else:
        ph3 = ph2 * ph
        ph4 = ph3 * ph
        ph5 = ph4 * ph
        cph = math.cos(ph)
        sph = math.sin(ph)
        m2 = (ph - sph) / ph3
        m3 = (1.0 - 0.5 * ph2 - cph) / ph4
        m4 = 0.5 * (m3 - 3.0 * (ph - sph - ph3 / 6.0) / ph5)

    t1 = 0.5 * rx
    t2 = m2 * (np.dot(px, rx) + np.dot(rx, px) + np.dot(px, np.dot(rx, px)))
    t3 = -m3 * (
        np.dot(px, np.dot(px, rx))
        + np.dot(rx, np.dot(px, px))
        - 3 * np.dot(px, np.dot(rx, px))
    )
    t4 = -m4 * (
        np.dot(px, np.dot(rx, np.dot(px, px)))
        + np.dot(px, np.dot(px, np.dot(rx, px)))
    )

    Q = t1 + t2 + t3 + t4

    return Q


def left_jacobian_SE3_inv_series(Stheta, n_terms):
    """Left inverse Jacobian of SE(3) at theta from Taylor series.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.

    n_terms : int
        Number of terms to include in the series.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SE(3).

    See Also
    --------
    left_jacobian_SE3_inv : Left inverse Jacobian of SE(3).
    """
    from scipy.special import bernoulli

    Stheta = check_exponential_coordinates(Stheta)
    J_inv = np.eye(6)
    pxn = np.eye(6)
    px = _curlyhat(Stheta)
    b = bernoulli(n_terms + 1)
    for n in range(n_terms):
        pxn = np.dot(pxn, px / (n + 1))
        J_inv += b[n + 1] * pxn
    return J_inv


def _curlyhat(Stheta):
    omega_matrix = cross_product_matrix(Stheta[:3])
    return np.block(
        [
            [omega_matrix, np.zeros((3, 3))],
            [cross_product_matrix(Stheta[3:]), omega_matrix],
        ]
    )
