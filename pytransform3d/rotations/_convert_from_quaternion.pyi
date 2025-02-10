import numpy as np
import numpy.typing as npt


def matrix_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def compact_axis_angle_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...


def mrp_from_quaternion(q: npt.ArrayLike) -> np.ndarray: ...
