import numpy as np
import numpy.typing as npt
from typing import Union


def matrix_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def matrix_power(R: npt.ArrayLike, t: float) -> np.ndarray: ...


def axis_angle_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def quaternion_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float,
        shortest_path: bool = ...) -> np.ndarray: ...


def rotor_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float,
        shortest_path: bool = ...) -> np.ndarray: ...


def slerp_weights(
        angle: float,
        t: Union[float, npt.ArrayLike]) -> Union[float, np.ndarray]: ...
