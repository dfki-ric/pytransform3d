import numpy as np
import numpy.typing as npt

def norm_euler(e: npt.ArrayLike, i: int, j: int, k: int) -> np.ndarray: ...
def euler_near_gimbal_lock(
    e: npt.ArrayLike, i: int, j: int, k: int, tolerance: float = ...
) -> bool: ...
def assert_euler_equal(
    e1: npt.ArrayLike,
    e2: npt.ArrayLike,
    i: int,
    j: int,
    k: int,
    *args,
    **kwargs,
): ...
def matrix_from_euler(
    e: npt.ArrayLike, i: int, j: int, k: int, extrinsic: bool
) -> np.ndarray: ...
def general_intrinsic_euler_from_active_matrix(
    R: npt.ArrayLike,
    n1: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    proper_euler: bool,
    strict_check: bool = ...,
) -> np.ndarray: ...
def euler_from_matrix(
    R: npt.ArrayLike,
    i: int,
    j: int,
    k: int,
    extrinsic: bool,
    strict_check: bool = ...,
) -> np.ndarray: ...
def euler_from_quaternion(
    q: npt.ArrayLike, i: int, j: int, k: int, extrinsic: bool
) -> np.ndarray: ...
