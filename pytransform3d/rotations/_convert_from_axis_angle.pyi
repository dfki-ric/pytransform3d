import numpy as np
import numpy.typing as npt


def matrix_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_compact_axis_angle(a: npt.ArrayLike) -> np.ndarray: ...
