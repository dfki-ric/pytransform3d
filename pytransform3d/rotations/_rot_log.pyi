import numpy as np
import numpy.typing as npt


def check_skew_symmetric_matrix(V: npt.ArrayLike, tolerance: float = ..., strict_check: bool = ...) -> np.ndarray: ...


def check_rot_log(V: npt.ArrayLike, tolerance: float = ..., strict_check: bool = ...) -> np.ndarray: ...


def cross_product_matrix(v: npt.ArrayLike) -> np.ndarray: ...


def rot_log_from_compact_axis_angle(v: npt.ArrayLike) -> np.ndarray: ...
