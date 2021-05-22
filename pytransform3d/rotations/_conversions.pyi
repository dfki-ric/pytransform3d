import numpy as np
import numpy.typing as npt


def cross_product_matrix(v: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_two_vectors(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_angle(basis: int, angle: float) -> np.ndarray: ...


def active_matrix_from_angle(basis: int, angle: float) -> np.ndarray: ...


def matrix_from_euler_xyz(e: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_euler_zyx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_xzx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_xzx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_xyx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_xyx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_yxy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_yxy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_yzy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_yzy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_zyz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_zyz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_zxz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_zxz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_xzy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_xzy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_xyz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_xyz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_yxz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_yxz(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_yzx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_yzx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_zyx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_zyx(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_intrinsic_euler_zxy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_euler_zxy(e: npt.ArrayLike) -> np.ndarray: ...


def active_matrix_from_extrinsic_roll_pitch_yaw(rpy: npt.ArrayLike) -> np.ndarray: ...


def _general_intrinsic_euler_from_active_matrix(
        R: npt.ArrayLike, n1: np.ndarray, n2: np.ndarray, n3: np.ndarray, proper_euler: bool, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_xzx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_xzx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_xyx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_xyx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_yxy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_yxy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_yzy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_yzy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_zyz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_zyz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_zxz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_zxz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_xzy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_xzy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_xyz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_xyz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_yxz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_yxz_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_yzx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_yzx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_zyx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_zyx_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def intrinsic_euler_zxy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def extrinsic_euler_zxy_from_active_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def axis_angle_from_matrix(R: npt.ArrayLike, strict_check: bool = ..., check: bool= ...) -> np.ndarray: ...


def axis_angle_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_two_directions(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def compact_axis_angle_from_matrix(R: npt.ArrayLike) -> np.ndarray: ...


def compact_axis_angle_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def quaternion_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def quaternion_xyzw_from_wxyz(q_wxyz: npt.ArrayLike) -> np.ndarray: ...


def quaternion_wxyz_from_xyzw(q_xyzw: npt.ArrayLike) -> np.ndarray: ...
