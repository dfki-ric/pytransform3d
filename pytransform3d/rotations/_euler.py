"""Euler angles."""

import math

import numpy as np
from numpy.testing import assert_array_almost_equal

from ._angle import norm_angle, active_matrix_from_angle
from ._constants import half_pi, unitx, unity, unitz, eps
from ._matrix import check_matrix
from ._quaternion import check_quaternion
from ._utils import check_axis_index


def norm_euler(e, i, j, k):
    """Normalize Euler angle range.

    Parameters
    ----------
    e : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    Returns
    -------
    e : array, shape (3,)
        Extracted rotation angles in radians about the axes i, j, k in this
        order. The first and last angle are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    alpha, beta, gamma = norm_angle(e)

    proper_euler = i == k
    if proper_euler:
        if beta < 0.0:
            alpha += np.pi
            beta *= -1.0
            gamma -= np.pi
    elif abs(beta) > half_pi:
        alpha += np.pi
        beta = np.pi - beta
        gamma -= np.pi

    return norm_angle([alpha, beta, gamma])


def euler_near_gimbal_lock(e, i, j, k, tolerance=1e-6):
    """Check if Euler angles are close to gimbal lock.

    Parameters
    ----------
    e : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    tolerance : float
        Tolerance for the comparison.

    Returns
    -------
    near_gimbal_lock : bool
        Indicates if the Euler angles are near the gimbal lock singularity.
    """
    e = norm_euler(e, i, j, k)
    beta = e[1]
    proper_euler = i == k
    if proper_euler:
        return abs(beta) < tolerance or abs(beta - np.pi) < tolerance
    else:
        return abs(abs(beta) - half_pi) < tolerance


def assert_euler_equal(e1, e2, i, j, k, *args, **kwargs):
    """Raise an assertion if two Euler angles are not approximately equal.

    Parameters
    ----------
    e1 : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    e2 : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    e1 = norm_euler(e1, i, j, k)
    e2 = norm_euler(e2, i, j, k)
    assert_array_almost_equal(e1, e2, *args, **kwargs)


def matrix_from_euler(e, i, j, k, extrinsic):
    """General method to compute active rotation matrix from any Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    Returns
    -------
    R : array, shape (3, 3)
        Active rotation matrix

    Raises
    ------
    ValueError
        If basis is invalid
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    alpha, beta, gamma = e
    if not extrinsic:
        i, k = k, i
        alpha, gamma = gamma, alpha
    R = (
        active_matrix_from_angle(k, gamma)
        .dot(active_matrix_from_angle(j, beta))
        .dot(active_matrix_from_angle(i, alpha))
    )
    return R


def general_intrinsic_euler_from_active_matrix(
    R, n1, n2, n3, proper_euler, strict_check=True
):
    """General algorithm to extract intrinsic euler angles from a matrix.

    The implementation is based on SciPy's implementation:
    https://github.com/scipy/scipy/blob/743c283bbe79473a03ca2eddaa537661846d8a19/scipy/spatial/transform/_rotation.pyx

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Active rotation matrix

    n1 : array, shape (3,)
        First rotation axis (basis vector)

    n2 : array, shape (3,)
        Second rotation axis (basis vector)

    n3 : array, shape (3,)
        Third rotation axis (basis vector)

    proper_euler : bool
        Is this an Euler angle convention or a Cardan / Tait-Bryan convention?
        Proper Euler angles rotate about the same axis twice, for example,
        z, y', and z''.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    euler_angles : array, shape (3,)
        Extracted intrinsic rotation angles in radians about the axes
        n1, n2, and n3 in this order. The first and last angle are
        normalized to [-pi, pi]. The middle angle is normalized to
        either [0, pi] (proper Euler angles) or [-pi/2, pi/2]
        (Cardan / Tait-Bryan angles).

    References
    ----------
    .. [1] Shuster, M. D., Markley, F. L. (2006).
       General Formula for Extracting the Euler Angles.
       Journal of Guidance, Control, and Dynamics, 29(1), pp 2015-221,
       doi: 10.2514/1.16622. https://arc.aiaa.org/doi/abs/10.2514/1.16622
    """
    D = check_matrix(R, strict_check=strict_check)

    # Differences to the paper:
    # - we call the angles alpha, beta, and gamma
    # - we obtain angles from intrinsic rotations, thus some matrices are
    #   transposed like in SciPy's implementation

    # Step 2
    # - Equation 5
    n1_cross_n2 = np.cross(n1, n2)
    lmbda = np.arctan2(np.dot(n1_cross_n2, n3), np.dot(n1, n3))
    # - Equation 6
    C = np.vstack((n2, n1_cross_n2, n1))

    # Step 3
    # - Equation 8
    CDCT = np.dot(np.dot(C, D), C.T)
    O = np.dot(CDCT, active_matrix_from_angle(0, lmbda).T)

    # Step 4
    # Fix numerical issue if O_22 is slightly out of range of arccos
    O_22 = max(min(O[2, 2], 1.0), -1.0)
    # - Equation 10a
    beta = lmbda + np.arccos(O_22)

    safe1 = abs(beta - lmbda) >= np.finfo(float).eps
    safe2 = abs(beta - lmbda - np.pi) >= np.finfo(float).eps
    if safe1 and safe2:  # Default case, no gimbal lock
        # Step 5
        # - Equation 10b
        alpha = np.arctan2(O[0, 2], -O[1, 2])
        # - Equation 10c
        gamma = np.arctan2(O[2, 0], O[2, 1])

        # Step 7
        if proper_euler:
            valid_beta = 0.0 <= beta <= np.pi
        else:  # Cardan / Tait-Bryan angles
            valid_beta = -0.5 * np.pi <= beta <= 0.5 * np.pi
        # - Equation 12
        if not valid_beta:
            alpha += np.pi
            beta = 2.0 * lmbda - beta
            gamma -= np.pi
    else:
        # Step 6 - Handle gimbal locks
        # a)
        gamma = 0.0
        if not safe1:
            # b)
            alpha = np.arctan2(O[1, 0] - O[0, 1], O[0, 0] + O[1, 1])
        else:
            # c)
            alpha = np.arctan2(O[1, 0] + O[0, 1], O[0, 0] - O[1, 1])
    euler_angles = norm_angle([alpha, beta, gamma])
    return euler_angles


def euler_from_matrix(R, i, j, k, extrinsic, strict_check=True):
    """General method to extract any Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Active rotation matrix

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    euler_angles : array, shape (3,)
        Extracted rotation angles in radians about the axes i, j, k in this
        order. The first and last angle are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).

    Raises
    ------
    ValueError
        If basis is invalid

    References
    ----------
    .. [1] Shuster, M. D., Markley, F. L. (2006).
       General Formula for Extracting the Euler Angles.
       Journal of Guidance, Control, and Dynamics, 29(1), pp 2015-221,
       doi: 10.2514/1.16622. https://arc.aiaa.org/doi/abs/10.2514/1.16622
    """
    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    basis_vectors = [unitx, unity, unitz]
    proper_euler = i == k
    if extrinsic:
        i, k = k, i
    e = general_intrinsic_euler_from_active_matrix(
        R,
        basis_vectors[i],
        basis_vectors[j],
        basis_vectors[k],
        proper_euler,
        strict_check,
    )

    if extrinsic:
        e = e[::-1]

    return e


def euler_from_quaternion(q, i, j, k, extrinsic):
    """General method to extract any Euler angles from quaternions.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    extrinsic : bool
        Do we use extrinsic transformations? Intrinsic otherwise.

    Returns
    -------
    euler_angles : array, shape (3,)
        Extracted rotation angles in radians about the axes i, j, k in this
        order. The first and last angle are normalized to [-pi, pi]. The middle
        angle is normalized to either [0, pi] (proper Euler angles) or
        [-pi/2, pi/2] (Cardan / Tait-Bryan angles).

    Raises
    ------
    ValueError
        If basis is invalid

    References
    ----------
    .. [1] Bernardes, E., Viollet, S. (2022). Quaternion to Euler angles
       conversion: A direct, general and computationally efficient method.
       PLOS ONE, 17(11), doi: 10.1371/journal.pone.0276302.
    """
    q = check_quaternion(q)

    check_axis_index("i", i)
    check_axis_index("j", j)
    check_axis_index("k", k)

    i += 1
    j += 1
    k += 1

    # The original algorithm assumes extrinsic convention. Hence, we swap
    # the order of axes for intrinsic rotation.
    if not extrinsic:
        i, k = k, i

    # Proper Euler angles rotate about the same axis in the first and last
    # rotation. If this is not the case, they are called Cardan or Tait-Bryan
    # angles and have to be handled differently.
    proper_euler = i == k
    if proper_euler:
        k = 6 - i - j

    sign = (i - j) * (j - k) * (k - i) // 2
    a = q[0]
    b = q[i]
    c = q[j]
    d = q[k] * sign

    if not proper_euler:
        a, b, c, d = a - c, b + d, c + a, d - b

    # Equation 34 is used instead of Equation 35 as atan2 it is numerically
    # more accurate than acos.
    angle_j = 2.0 * math.atan2(math.hypot(c, d), math.hypot(a, b))

    # Check for singularities
    if abs(angle_j) <= eps:
        singularity = 1
    elif abs(angle_j - math.pi) <= eps:
        singularity = 2
    else:
        singularity = 0

    # Equation 25
    # (theta_1 + theta_3) / 2
    half_sum = math.atan2(b, a)
    # (theta_1 - theta_3) / 2
    half_diff = math.atan2(d, c)

    if singularity == 0:  # no singularity
        # Equation 32
        angle_i = half_sum + half_diff
        angle_k = half_sum - half_diff
    elif extrinsic:  # singularity
        angle_k = 0.0
        if singularity == 1:
            angle_i = 2.0 * half_sum
        else:
            assert singularity == 2
            angle_i = 2.0 * half_diff
    else:  # intrinsic, singularity
        angle_i = 0.0
        if singularity == 1:
            angle_k = 2.0 * half_sum
        else:
            assert singularity == 2
            angle_k = -2.0 * half_diff

    if not proper_euler:
        # Equation 43
        angle_j -= math.pi / 2.0
        # Equation 44
        angle_i *= sign

    angle_k = norm_angle(angle_k)
    angle_i = norm_angle(angle_i)

    if extrinsic:
        return np.array([angle_k, angle_j, angle_i])

    return np.array([angle_i, angle_j, angle_k])
