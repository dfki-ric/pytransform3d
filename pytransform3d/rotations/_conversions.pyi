import numpy as np
import numpy.typing as npt


def cross_product_matrix(v: npt.ArrayLike) -> np.ndarray: ...


def rot_log_from_compact_axis_angle(v: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_matrix(R: npt.ArrayLike, strict_check: bool = ..., check: bool= ...) -> np.ndarray: ...


def compact_axis_angle_from_matrix(R: npt.ArrayLike, check: bool = ...) -> np.ndarray: ...
