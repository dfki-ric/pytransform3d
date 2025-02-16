import numpy as np
import numpy.typing as npt

def norm_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def norm_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def compact_axis_angle_near_pi(
    a: npt.ArrayLike, tolerance: float = ...
) -> bool: ...
def check_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def check_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def assert_axis_angle_equal(
    a1: npt.ArrayLike, a2: npt.ArrayLike, *args, **kwargs
): ...
def assert_compact_axis_angle_equal(
    a1: npt.ArrayLike, a2: npt.ArrayLike, *args, **kwargs
): ...
def axis_angle_from_two_directions(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> np.ndarray: ...
def matrix_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def matrix_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def axis_angle_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def quaternion_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def quaternion_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
def mrp_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
