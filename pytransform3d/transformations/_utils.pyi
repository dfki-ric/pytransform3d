import numpy as np
import numpy.typing as npt
from typing import Tuple


def check_transform(
        A2B: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def check_pq(pq: npt.ArrayLike) -> np.ndarray: ...


def check_screw_parameters(
        q: npt.ArrayLike, s_axis: npt.ArrayLike,
        h: float) -> Tuple[np.ndarray, np.ndarray, float]: ...


def check_screw_axis(screw_axis: npt.ArrayLike) -> np.ndarray: ...


def check_exponential_coordinates(Stheta: npt.ArrayLike) -> np.ndarray: ...


def check_screw_matrix(screw_matrix: npt.ArrayLike, tolerance: float = ...,
                       strict_check: bool = ...) -> np.ndarray: ...


def check_transform_log(transform_log: npt.ArrayLike, tolerance: float = ...,
                        strict_check: bool = ...) -> np.ndarray: ...


def check_dual_quaternion(dq: npt.ArrayLike, unit: bool = ...) -> np.ndarray: ...
