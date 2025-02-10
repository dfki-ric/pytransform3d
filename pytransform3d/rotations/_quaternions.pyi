import numpy as np
import numpy.typing as npt


def quaternion_requires_renormalization(q: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def check_quaternion(q: npt.ArrayLike, unit: bool = ...) -> np.ndarray: ...


def check_quaternions(Q: npt.ArrayLike, unit: bool = ...) -> np.ndarray: ...


def quaternion_double(q: npt.ArrayLike) -> np.ndarray: ...


def pick_closest_quaternion(
        quaternion: npt.ArrayLike,
        target_quaternion: npt.ArrayLike) -> np.ndarray: ...


def pick_closest_quaternion_impl(
        quaternion: np.ndarray, target_quaternion: np.ndarray
) -> np.ndarray: ...


def quaternion_integrate(Qd: npt.ArrayLike, q0: npt.ArrayLike = ..., dt: float = ...) -> np.ndarray: ...


def quaternion_gradient(Q: npt.ArrayLike, dt: float = ...) -> np.ndarray: ...


def concatenate_quaternions(q1: npt.ArrayLike, q2: npt.ArrayLike) -> np.ndarray: ...


def q_prod_vector(q: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray: ...


def q_conj(q: npt.ArrayLike) -> np.ndarray: ...


def quaternion_dist(q1: npt.ArrayLike, q2: npt.ArrayLike) -> np.ndarray: ...


def quaternion_diff(q1: npt.ArrayLike, q2: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_euler(
        e: npt.ArrayLike, i: int, j: int, k: int, extrinsic: bool) -> np.ndarray: ...


def matrix_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def compact_axis_angle_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def mrp_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def quaternion_xyzw_from_wxyz(q_wxyz: npt.ArrayLike) -> np.ndarray: ...


def quaternion_wxyz_from_xyzw(q_xyzw: npt.ArrayLike) -> np.ndarray: ...
