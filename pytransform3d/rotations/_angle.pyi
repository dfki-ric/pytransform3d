import numpy as np
import numpy.typing as npt


def norm_angle(a: npt.ArrayLike) -> np.ndarray: ...


def passive_matrix_from_angle(basis: int, angle: float) -> np.ndarray: ...


def active_matrix_from_angle(basis: int, angle: float) -> np.ndarray: ...


def quaternion_from_angle(basis: int, angle: float) -> np.ndarray: ...
