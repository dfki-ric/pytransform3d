import numpy as np
import numpy.typing as npt


def check_pq(pq: npt.ArrayLike) -> np.ndarray: ...


def pq_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t:float) -> np.ndarray: ...


def transform_from_pq(pq: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternion_from_pq(pq: npt.ArrayLike) -> np.ndarray: ...
