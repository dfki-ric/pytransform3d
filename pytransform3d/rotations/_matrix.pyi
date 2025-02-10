import numpy as np
import numpy.typing as npt


def matrix_from_two_vectors(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_matrix(R: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def axis_angle_from_matrix(R: npt.ArrayLike, strict_check: bool = ..., check: bool= ...) -> np.ndarray: ...


def compact_axis_angle_from_matrix(R: npt.ArrayLike, check: bool = ...) -> np.ndarray: ...
