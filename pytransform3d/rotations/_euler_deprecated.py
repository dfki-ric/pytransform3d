import warnings

from ._angle import active_matrix_from_angle
from ._constants import unitx, unity, unitz
from ._euler import general_intrinsic_euler_from_active_matrix
from ._matrix import quaternion_from_matrix


def active_matrix_from_intrinsic_euler_xzx(e):
    """Compute active rotation matrix from intrinsic xzx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, alpha)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(0, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_xzx(e):
    """Compute active rotation matrix from extrinsic xzx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, gamma)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(0, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_xyx(e):
    """Compute active rotation matrix from intrinsic xyx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, alpha)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(0, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_xyx(e):
    """Compute active rotation matrix from extrinsic xyx Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, gamma)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(0, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_yxy(e):
    """Compute active rotation matrix from intrinsic yxy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, alpha)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(1, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_yxy(e):
    """Compute active rotation matrix from extrinsic yxy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, gamma)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(1, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_yzy(e):
    """Compute active rotation matrix from intrinsic yzy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, alpha)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(1, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_yzy(e):
    """Compute active rotation matrix from extrinsic yzy Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, gamma)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(1, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_zyz(e):
    """Compute active rotation matrix from intrinsic zyz Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, alpha)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(2, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_zyz(e):
    """Compute active rotation matrix from extrinsic zyz Euler angles.

    .. warning::

        This function was not implemented correctly in versions 1.3 and 1.4
        as the order of the angles was reversed, which actually corresponds
        to intrinsic rotations. This has been fixed in version 1.5.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, gamma)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(2, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_zxz(e):
    """Compute active rotation matrix from intrinsic zxz Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, alpha)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(2, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_zxz(e):
    """Compute active rotation matrix from extrinsic zxz Euler angles.

    .. warning::

        This function was not implemented correctly in versions 1.3 and 1.4
        as the order of the angles was reversed, which actually corresponds
        to intrinsic rotations. This has been fixed in version 1.5.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, gamma)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(2, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_xzy(e):
    """Compute active rotation matrix from intrinsic xzy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, alpha)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(1, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_xzy(e):
    """Compute active rotation matrix from extrinsic xzy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, z-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, gamma)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(0, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_xyz(e):
    """Compute active rotation matrix from intrinsic xyz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, alpha)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(2, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_xyz(e):
    """Compute active rotation matrix from extrinsic xyz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, gamma)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(0, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_yxz(e):
    """Compute active rotation matrix from intrinsic yxz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x'-, and z''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, alpha)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(2, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_yxz(e):
    """Compute active rotation matrix from extrinsic yxz Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, x-, and z-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, gamma)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(1, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_yzx(e):
    """Compute active rotation matrix from intrinsic yzx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, alpha)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(0, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_yzx(e):
    """Compute active rotation matrix from extrinsic yzx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around y-, z-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, gamma)
        .dot(active_matrix_from_angle(2, beta))
        .dot(active_matrix_from_angle(1, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_zyx(e):
    """Compute active rotation matrix from intrinsic zyx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, alpha)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(0, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_zyx(e):
    """Compute active rotation matrix from extrinsic zyx Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y-, and x-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(0, gamma)
        .dot(active_matrix_from_angle(1, beta))
        .dot(active_matrix_from_angle(2, alpha))
    )
    return R


def active_matrix_from_intrinsic_euler_zxy(e):
    """Compute active rotation matrix from intrinsic zxy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x'-, and y''-axes (intrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(2, alpha)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(1, gamma))
    )
    return R


def active_matrix_from_extrinsic_euler_zxy(e):
    """Compute active rotation matrix from extrinsic zxy Cardan angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, x-, and y-axes (extrinsic rotations)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    R = (
        active_matrix_from_angle(1, gamma)
        .dot(active_matrix_from_angle(0, beta))
        .dot(active_matrix_from_angle(2, alpha))
    )
    return R


def active_matrix_from_extrinsic_roll_pitch_yaw(rpy):
    """Compute active rotation matrix from extrinsic roll, pitch, and yaw.

    Parameters
    ----------
    rpy : array-like, shape (3,)
        Angles for rotation around x- (roll), y- (pitch), and z-axes (yaw),
        extrinsic rotations

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    return active_matrix_from_extrinsic_euler_xyz(rpy)


def intrinsic_euler_xzx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xzx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z'-, and x''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unitx, True, strict_check
    )


def extrinsic_euler_xzx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xzx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z-, and x-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unitx, True, strict_check
    )[::-1]


def intrinsic_euler_xyx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xyx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y'-, and x''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitx, True, strict_check
    )


def extrinsic_euler_xyx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xyx Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y-, and x-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitx, True, strict_check
    )[::-1]


def intrinsic_euler_yxy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yxy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x'-, and y''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unity, True, strict_check
    )


def extrinsic_euler_yxy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yxy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x-, and y-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unity, True, strict_check
    )[::-1]


def intrinsic_euler_yzy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yzy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z'-, and y''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unity, True, strict_check
    )


def extrinsic_euler_yzy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yzy Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z-, and y-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unity, True, strict_check
    )[::-1]


def intrinsic_euler_zyz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zyz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y'-, and z''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitz, True, strict_check
    )


def extrinsic_euler_zyz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zyz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y-, and z-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitz, True, strict_check
    )[::-1]


def intrinsic_euler_zxz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zxz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x'-, and z''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unitz, True, strict_check
    )


def extrinsic_euler_zxz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zxz Euler angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x-, and z-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unitz, True, strict_check
    )[::-1]


def intrinsic_euler_xzy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xzy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z'-, and y''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unity, False, strict_check
    )


def extrinsic_euler_xzy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xzy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, z-, and y-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unitx, False, strict_check
    )[::-1]


def intrinsic_euler_xyz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic xyz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitz, False, strict_check
    )


def extrinsic_euler_xyz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic xyz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitx, False, strict_check
    )[::-1]


def intrinsic_euler_yxz_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yxz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x'-, and z''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unitz, False, strict_check
    )


def extrinsic_euler_yxz_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yxz Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, x-, and z-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unity, False, strict_check
    )[::-1]


def intrinsic_euler_yzx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic yzx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z'-, and x''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitz, unitx, False, strict_check
    )


def extrinsic_euler_yzx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic yzx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around y-, z-, and x-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unitz, unity, False, strict_check
    )[::-1]


def intrinsic_euler_zyx_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zyx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unity, unitx, False, strict_check
    )


def extrinsic_euler_zyx_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zyx Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, y-, and x-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitx, unity, unitz, False, strict_check
    )[::-1]


def intrinsic_euler_zxy_from_active_matrix(R, strict_check=True):
    """Compute intrinsic zxy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x'-, and y''-axes (intrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unitz, unitx, unity, False, strict_check
    )


def extrinsic_euler_zxy_from_active_matrix(R, strict_check=True):
    """Compute extrinsic zxy Cardan angles from active rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    strict_check : bool, optional (default: True)
        Raise a ValueError if the rotation matrix is not numerically close
        enough to a real rotation matrix. Otherwise we print a warning.

    Returns
    -------
    e : array, shape (3,)
        Angles for rotation around z-, x-, and y-axes (extrinsic rotations)
    """
    return general_intrinsic_euler_from_active_matrix(
        R, unity, unitx, unitz, False, strict_check
    )[::-1]


def quaternion_from_extrinsic_euler_xyz(e):
    """Compute quaternion from extrinsic xyz Euler angles.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y-, and z-axes (extrinsic rotations)

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    warnings.warn(
        "quaternion_from_extrinsic_euler_xyz is deprecated, use "
        "quaternion_from_euler",
        DeprecationWarning,
        stacklevel=2,
    )
    R = active_matrix_from_extrinsic_euler_xyz(e)
    return quaternion_from_matrix(R)
