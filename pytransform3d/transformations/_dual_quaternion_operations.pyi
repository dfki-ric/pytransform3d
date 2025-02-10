import numpy as np
import numpy.typing as npt


def dual_quaternion_requires_renormalization(
        dq: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def norm_dual_quaternion(dq: npt.ArrayLike) -> np.ndarray: ...


def check_dual_quaternion(
        dq: npt.ArrayLike, unit: bool = ...) -> np.ndarray: ...


def dual_quaternion_double(dq: npt.ArrayLike) -> np.ndarray: ...


def dq_conj(dq: npt.ArrayLike) -> np.ndarray: ...


def dq_q_conj(dq: npt.ArrayLike) -> np.ndarray: ...


def concatenate_dual_quaternions(
        dq1: npt.ArrayLike, dq2: npt.ArrayLike,
        unit: bool = ...) -> np.ndarray: ...


def dq_prod_vector(dq: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternion_sclerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def dual_quaternion_power(dq: npt.ArrayLike, t: float) -> np.ndarray: ...


def screw_parameters_from_dual_quaternion(dq: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternion_from_screw_parameters(
        q: npt.ArrayLike, s_axis: npt.ArrayLike, h: float,
        theta: float) -> np.ndarray: ...
