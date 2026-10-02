"""Jacobians of SO(3)."""

import math

import numpy as np

from ._rot_log import cross_product_matrix


def left_jacobian_SO3(omega):
    r"""Left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \boldsymbol{J}(\theta)
        =
        \frac{\sin{\theta}}{\theta} \boldsymbol{I}
        + \left(\frac{1 - \cos{\theta}}{\theta}\right)
        \left[\hat{\boldsymbol{\omega}}\right]
        + \left(1 - \frac{\sin{\theta}}{\theta} \right)
        \hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    Returns
    -------
    J : array, shape (3, 3)
        Left Jacobian of SO(3).

    See also
    --------
    left_jacobian_SO3_series :
        Left Jacobian of SO(3) at theta from Taylor series.

    left_jacobian_SO3_inv :
        Inverse left Jacobian of SO(3) at theta (angle of rotation).
    """
    omega = np.asarray(omega)
    theta = np.linalg.norm(omega)
    # theta is the rotation angle in radians. Float64 machine epsilon
    # eps = np.finfo(float).eps is one ULP above 1.0 (unit in the last
    # place, i.e. the spacing between adjacent floats).
    # The coefficient 1 - sin(theta)/theta = theta**2/6 + O(theta**4)
    # subtracts values near 1.0, losing relative precision as theta**2
    # approaches eps. The inverse coefficient starts with theta**2/12;
    # its conservative sqrt(6*eps) cutoff puts that leading term at eps/2.
    # Mirror this cutoff so both Jacobians use their series, avoiding
    # these scalar subtractions in the same tiny-angle range.
    if theta < math.sqrt(6.0 * np.finfo(float).eps):
        return left_jacobian_SO3_series(omega, 10)
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        # This coefficient is (1 - cos(theta))/theta. The half-angle
        # identity 1 - cos(theta) = 2*sin(theta/2)**2 avoids subtracting
        # two values near 1.0. For theta << 1 rad, the difference is about
        # theta**2/2, so O(eps) rounding in cos causes O(eps/theta**2)
        # relative error, even above the series cutoff; around sqrt(eps)
        # it can round to zero. sin(theta/2) is instead about theta/2 and
        # preserves the small value without this subtraction.
        + 2.0 * math.sin(0.5 * theta) ** 2 / theta * omega_matrix
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

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).
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
    r"""Inverse left Jacobian of SO(3) at theta (angle of rotation).

    .. math::

        \boldsymbol{J}^{-1}(\theta)
        =
        \frac{\theta}{2 \tan{\frac{\theta}{2}}} \boldsymbol{I}
        - \frac{\theta}{2} \left[\hat{\boldsymbol{\omega}}\right]
        + \left(1 - \frac{\theta}{2 \tan{\frac{\theta}{2}}}\right)
        \hat{\boldsymbol{\omega}} \hat{\boldsymbol{\omega}}^T

    Parameters
    ----------
    omega : array-like, shape (3,)
        Compact axis-angle representation.

    Returns
    -------
    J_inv : array, shape (3, 3)
        Inverse left Jacobian of SO(3).

    See Also
    --------
    left_jacobian_SO3 : Left Jacobian of SO(3) at theta (angle of rotation).

    left_jacobian_SO3_inv_series :
        Inverse left Jacobian of SO(3) at theta from Taylor series.
    """
    omega = np.asarray(omega)
    theta = np.linalg.norm(omega)
    # theta is the rotation angle in radians; eps is float64 machine
    # epsilon. The coefficient 1 - theta/(2*tan(theta/2)) starts with
    # theta**2/12. At theta = sqrt(6*eps), this leading correction is
    # eps/2, comparable to roundoff in the values near 1.0 being subtracted.
    # The difference can lose relative precision or round to zero.
    # Use the series below this conservative cutoff.
    if theta < math.sqrt(6.0 * np.finfo(float).eps):
        return left_jacobian_SO3_inv_series(omega, 10)
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        - 0.5 * omega_matrix * theta
        + (1.0 - 0.5 * theta / np.tan(theta / 2.0))
        * np.dot(omega_matrix, omega_matrix)
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

    See Also
    --------
    left_jacobian_SO3_inv :
        Inverse left Jacobian of SO(3) at theta (angle of rotation).
    """
    from scipy.special import bernoulli

    omega = np.asarray(omega)
    J_inv = np.eye(3)
    pxn = np.eye(3)
    px = cross_product_matrix(omega)
    b = bernoulli(n_terms + 1)
    for n in range(n_terms):
        pxn = np.dot(pxn, px / (n + 1))
        J_inv += b[n + 1] * pxn
    return J_inv
