import numpy as np
import numpy.typing as npt

def check_matrix(
    R: npt.ArrayLike, tolerance: float = ..., strict_check: bool = ...
) -> np.ndarray: ...
def matrix_requires_renormalization(
    R: npt.ArrayLike, tolerance: float = ...
) -> bool: ...
def norm_matrix(R: npt.ArrayLike) -> np.ndarray: ...
def assert_rotation_matrix(R: npt.ArrayLike, *args, **kwargs): ...
def matrix_from_two_vectors(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> np.ndarray: ...
def quaternion_from_matrix(
    R: npt.ArrayLike, strict_check: bool = ...
) -> np.ndarray: ...
def axis_angle_from_matrix(
    R: npt.ArrayLike, strict_check: bool = ..., check: bool = ...
) -> np.ndarray: ...
def compact_axis_angle_from_matrix(
    R: npt.ArrayLike, check: bool = ...
) -> np.ndarray: ...
