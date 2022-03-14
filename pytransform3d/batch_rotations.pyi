import numpy as np
import numpy.typing as npt
from typing import Union, AnyStr


def norm_vectors(
        V: npt.ArrayLike,
        out: Union[npt.ArrayLike, None] = ...) -> np.ndarray: ...


def angles_between_vectors(A: npt.ArrayLike, B: npt.ArrayLike) -> np.ndarray: ...


def active_matrices_from_angles(
        basis: int, angles: npt.ArrayLike,
        out: Union[npt.ArrayLike, None] = ...) -> np.ndarray: ...


def active_matrices_from_intrinsic_euler_angles(
        basis1: int, basis2: int, basis3: int, e: npt.ArrayLike,
        out: Union[npt.ArrayLike, None] = ...) -> np.ndarray: ...


def active_matrices_from_extrinsic_euler_angles(
        basis1: int, basis2: int, basis3: int, e: npt.ArrayLike,
        out: Union[npt.ArrayLike, None] = ...) -> np.ndarray: ...


def matrices_from_compact_axis_angles(
        A: Union[npt.ArrayLike, None] = ...,
        axes: Union[np.ndarray, None] = ...,
        angles: Union[np.ndarray, None] = ...,
        out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def axis_angles_from_matrices(
        Rs: npt.ArrayLike, traces: Union[np.ndarray, None] = ...,
        out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def cross_product_matrices(V: npt.ArrayLike) -> np.ndarray: ...


def matrices_from_quaternions(
        Q: npt.ArrayLike, normalize_quaternions: bool = ...,
        out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def quaternions_from_matrices(
        Rs: npt.ArrayLike,
        out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def quaternion_slerp_batch(
        start: npt.ArrayLike, end: npt.ArrayLike,
        t: npt.ArrayLike, shortest_path: bool = ...) -> np.ndarray: ...


def batch_concatenate_quaternions(
        Q1: npt.ArrayLike, Q2: npt.ArrayLike,
        out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def batch_q_conj(Q: npt.ArrayLike) -> np.ndarray: ...


def batch_quaternion_wxyz_from_xyzw(
    Q_xyzw: npt.ArrayLike,
    out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def batch_quaternion_xyzw_from_wxyz(
    Q_wxyz: npt.ArrayLike,
    out: Union[np.ndarray, None] = ...) -> np.ndarray: ...


def smooth_quaternion_trajectory(
    Q: npt.ArrayLike,
    start_component_positive: AnyStr = ...) -> np.ndarray: ...
