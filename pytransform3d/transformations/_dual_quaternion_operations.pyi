import numpy as np
import numpy.typing as npt


def dq_conj(dq: npt.ArrayLike) -> np.ndarray: ...


def dq_q_conj(dq: npt.ArrayLike) -> np.ndarray: ...


def concatenate_dual_quaternions(
        dq1: npt.ArrayLike, dq2: npt.ArrayLike) -> np.ndarray: ...


def dq_prod_vector(dq: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternion_sclerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def dual_quaternion_power(dq: npt.ArrayLike, t: float) -> np.ndarray: ...
