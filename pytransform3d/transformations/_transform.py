"""Transformation matrices."""

import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal

from ..rotations import (
    matrix_requires_renormalization,
    check_matrix,
    assert_rotation_matrix,
    quaternion_from_matrix,
    compact_axis_angle_from_matrix,
    left_jacobian_SO3_inv,
    cross_product_matrix,
    concatenate_quaternions,
)


def transform_requires_renormalization(A2B, tolerance=1e-6):
    r"""Check if transformation matrix requires renormalization.

    This function will check if :math:`R R^T \approx I`.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B with a rotation matrix that should
        be orthonormal.

    tolerance : float, optional (default: 1e-6)
        Tolerance for check.

    Returns
    -------
    required : bool
        Indicates if renormalization is required.

    See Also
    --------
    pytransform3d.rotations.matrix_requires_renormalization
        Check if a rotation matrix needs renormalization.
    pytransform3d.rotations.norm_matrix : Orthonormalize rotation matrix.
    """
    return matrix_requires_renormalization(np.asarray(A2B[:3, :3]), tolerance)


def check_transform(A2B, strict_check=True):
    """Input validation of transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (4, 4)
        Validated transform from frame A to frame B

    Raises
    ------
    ValueError
        If input is invalid
    """
    A2B = np.asarray(A2B, dtype=np.float64)
    if A2B.ndim != 2 or A2B.shape[0] != 4 or A2B.shape[1] != 4:
        raise ValueError(
            "Expected homogeneous transformation matrix with "
            "shape (4, 4), got array-like object with shape %s" % (A2B.shape,)
        )
    check_matrix(A2B[:3, :3], strict_check=strict_check)
    if not np.allclose(A2B[3], np.array([0.0, 0.0, 0.0, 1.0])):
        error_msg = (
            "Excpected homogeneous transformation matrix with "
            "[0, 0, 0, 1] at the bottom, got %r" % A2B
        )
        if strict_check:
            raise ValueError(error_msg)
        warnings.warn(error_msg)
    return A2B


def assert_transform(A2B, *args, **kwargs):
    """Raise an assertion if the transform is not a homogeneous matrix.

    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    assert_rotation_matrix(A2B[:3, :3], *args, **kwargs)
    assert_array_almost_equal(
        A2B[3], np.array([0.0, 0.0, 0.0, 1.0]), *args, **kwargs
    )


def transform_from(R, p, strict_check=True):
    r"""Make transformation from rotation matrix and translation.

    .. math::

        \boldsymbol{T}_{BA} = \left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right) \in SE(3)

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    A2B = rotate_transform(np.eye(4), R, strict_check=strict_check, check=False)
    A2B = translate_transform(A2B, p, strict_check=strict_check, check=False)
    return A2B


def translate_transform(A2B, p, strict_check=True, check=True):
    """Sets the translation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    p : array-like, shape (3,)
        Translation

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, -1] = p
    return out


def rotate_transform(A2B, R, strict_check=True, check=True):
    """Sets the rotation of a transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    out = A2B.copy()
    out[:3, :3] = R
    return out


def pq_from_transform(A2B, strict_check=True):
    """Compute position and quaternion from transformation matrix.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transformation matrix from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    pq : array, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    return np.hstack((A2B[:3, 3], quaternion_from_matrix(A2B[:3, :3])))


def transform_log_from_transform(A2B, strict_check=True):
    r"""Compute matrix logarithm of transformation from transformation.

    Logarithmic map.

    .. math::

        \log: \boldsymbol{T} \in SE(3)
        \rightarrow \left[ \mathcal{S} \right] \theta \in se(3)

    .. math::

        \log(\boldsymbol{T}) =
        \log\left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)
        =
        \left(
        \begin{array}{cc}
        \log\boldsymbol{R} & \boldsymbol{J}^{-1}(\theta) \boldsymbol{p}\\
        \boldsymbol{0} & 0
        \end{array}
        \right)
        =
        \left(
        \begin{array}{cc}
        \hat{\boldsymbol{\omega}} \theta
        & \boldsymbol{v} \theta\\
        \boldsymbol{0} & 0
        \end{array}
        \right)
        =
        \left[\mathcal{S}\right]\theta,

    where :math:`\boldsymbol{J}^{-1}(\theta)` is the inverse left Jacobian of
    :math:`SO(3)` (see :func:`~pytransform3d.rotations.left_jacobian_SO3_inv`).

    Parameters
    ----------
    A2B : array, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    transform_log : array, shape (4, 4)
        Matrix logarithm of transformation matrix: [S] * theta.
    """
    A2B = check_transform(A2B, strict_check=strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    transform_log = np.zeros((4, 4))

    if np.linalg.norm(np.eye(3) - R) < np.finfo(float).eps:
        transform_log[:3, 3] = p
        return transform_log

    omega_theta = compact_axis_angle_from_matrix(R)
    theta = np.linalg.norm(omega_theta)

    if theta == 0:
        return transform_log

    J_inv = left_jacobian_SO3_inv(omega_theta)
    v_theta = np.dot(J_inv, p)

    transform_log[:3, :3] = cross_product_matrix(omega_theta)
    transform_log[:3, 3] = v_theta

    return transform_log


def exponential_coordinates_from_transform(A2B, strict_check=True, check=True):
    r"""Compute exponential coordinates from transformation matrix.

    Logarithmic map.

    .. math::

        Log: \boldsymbol{T} \in SE(3)
        \rightarrow \mathcal{S} \theta \in \mathbb{R}^6

    .. math::

        Log(\boldsymbol{T}) =
        Log\left(
        \begin{array}{cc}
        \boldsymbol{R} & \boldsymbol{p}\\
        \boldsymbol{0} & 1
        \end{array}
        \right)
        =
        \left(
        \begin{array}{c}
        Log(\boldsymbol{R})\\
        \boldsymbol{J}^{-1}(\theta) \boldsymbol{p}
        \end{array}
        \right)
        =
        \left(
        \begin{array}{c}
        \hat{\boldsymbol{\omega}}\\
        \boldsymbol{v}
        \end{array}
        \right)
        \theta
        =
        \mathcal{S}\theta,

    where :math:`\boldsymbol{J}^{-1}(\theta)` is the inverse left Jacobian of
    :math:`SO(3)` (see :func:`~pytransform3d.rotations.left_jacobian_SO3_inv`).

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transformation matrix from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    Stheta : array, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where S is the screw axis, the first 3 components are related to
        rotation and the last 3 components are related to translation.
        Theta is the rotation angle and h * theta the translation.
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    if np.linalg.norm(np.eye(3) - R) < np.finfo(float).eps:
        return np.r_[0.0, 0.0, 0.0, p]

    omega_theta = compact_axis_angle_from_matrix(R, check=check)
    theta = np.linalg.norm(omega_theta)

    if theta == 0:
        return np.r_[0.0, 0.0, 0.0, p]

    v_theta = np.dot(left_jacobian_SO3_inv(omega_theta), p)

    return np.hstack((omega_theta, v_theta))


def dual_quaternion_from_transform(A2B):
    """Compute dual quaternion from transformation matrix.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    Returns
    -------
    dq : array, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    A2B = check_transform(A2B)
    real = quaternion_from_matrix(A2B[:3, :3])
    dual = 0.5 * concatenate_quaternions(np.r_[0, A2B[:3, 3]], real)
    return np.hstack((real, dual))


def adjoint_from_transform(A2B, strict_check=True, check=True):
    r"""Compute adjoint representation of a transformation matrix.

    The adjoint representation of a transformation
    :math:`\left[Ad_{\boldsymbol{T}_{BA}}\right] \in \mathbb{R}^{6 \times 6}`
    from frame A to frame B translates a twist from frame A to frame B
    through the adjoint map

    .. math::

        \mathcal{V}_{B}
        = \left[Ad_{\boldsymbol{T}_{BA}}\right] \mathcal{V}_A

    The corresponding transformation matrix operation is

    .. math::

        \left[\mathcal{V}_{B}\right]
        = \boldsymbol{T}_{BA} \left[\mathcal{V}_A\right]
        \boldsymbol{T}_{BA}^{-1}

    We can also use the adjoint representation to transform a wrench from frame
    A to frame B:

    .. math::

        \mathcal{F}_B
        = \left[ Ad_{\boldsymbol{T}_{AB}} \right]^T \mathcal{F}_A

    Note that not only the adjoint is transposed but also the transformation is
    inverted.

    Adjoint representations have the following properties:

    .. math::

        \left[Ad_{\boldsymbol{T}_1 \boldsymbol{T}_2}\right]
        = \left[Ad_{\boldsymbol{T}_1}\right]
        \left[Ad_{\boldsymbol{T}_2}\right]

    .. math::

        \left[Ad_{\boldsymbol{T}}\right]^{-1} =
        \left[Ad_{\boldsymbol{T}^{-1}}\right]

    For a transformation matrix

    .. math::

        \boldsymbol T =
        \left( \begin{array}{cc}
            \boldsymbol R & \boldsymbol t\\
            \boldsymbol 0 & 1\\
        \end{array} \right)

    the adjoint is defined as

    .. math::

        \left[Ad_{\boldsymbol{T}}\right]
        =
        \left( \begin{array}{cc}
            \boldsymbol R & \boldsymbol 0\\
            \left[\boldsymbol{t}\right]_{\times}\boldsymbol R & \boldsymbol R\\
        \end{array} \right),

    where :math:`\left[\boldsymbol{t}\right]_{\times}` is the cross-product
    matrix (see :func:`~pytransform3d.rotations.cross_product_matrix`) of the
    translation component.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    adj_A2B : array, shape (6, 6)
        Adjoint representation of transformation matrix
    """
    if check:
        A2B = check_transform(A2B, strict_check)

    R = A2B[:3, :3]
    p = A2B[:3, 3]

    adj_A2B = np.zeros((6, 6))
    adj_A2B[:3, :3] = R
    adj_A2B[3:, :3] = np.dot(cross_product_matrix(p), R)
    adj_A2B[3:, 3:] = R
    return adj_A2B
