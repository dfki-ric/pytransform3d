import numpy as np
import numpy.typing as npt
from typing import Tuple


def check_axis_index(name: str, i: int): ...


def norm_vector(v: npt.ArrayLike) -> np.ndarray: ...


def perpendicular_to_vectors(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def perpendicular_to_vector(a: npt.ArrayLike) -> np.ndarray: ...


def angle_between_vectors(a: npt.ArrayLike, b: npt.ArrayLike, fast: bool = ...) -> float: ...


def vector_projection(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def plane_basis_from_normal(
        plane_normal: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def check_matrix(R: npt.ArrayLike, tolerance: float = ..., strict_check: bool = ...) -> np.ndarray: ...
