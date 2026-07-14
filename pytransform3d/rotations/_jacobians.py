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
    # The closed-form terms (1 - cos(t))/t and 1 - sin(t)/t involve
    # subtracting a value near 1 from 1.  The leading correction is O(t)
    # and O(t^2) respectively.  When that correction is smaller than half
    # a ULP (unit in the last place, i.e. the spacing between adjacent
    # floats) of 1.0, the subtraction result is exactly 0 and all
    # significant bits are lost.  ULP(1.0) = eps = 2^-52 ~= 2.2e-16 for
    # float64.  The tighter condition, 1 - sin(t)/t ~= t^2/6 < eps/2,
    # gives t < sqrt(3*eps).  We use sqrt(6*eps) (~3.65e-8) as a
    # slightly conservative threshold that also covers the inverse.
    if theta < math.sqrt(6.0 * np.finfo(float).eps):
        return left_jacobian_SO3_series(omega, 10)
    omega_unit = omega / theta
    omega_matrix = cross_product_matrix(omega_unit)
    return (
        np.eye(3)
        + (1.0 - math.cos(theta)) / theta * omega_matrix
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
    # 1 - t/(2*tan(t/2)) ~= t^2/12 for small t.  This falls below half a
    # ULP (unit in the last place) of 1.0 for t < sqrt(6*eps), so the
    # subtraction yields exactly 0 in float64.  Use the series instead.
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
