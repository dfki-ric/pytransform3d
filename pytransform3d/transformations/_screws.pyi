import numpy as np
import numpy.typing as npt
from typing import Tuple


def norm_exponential_coordinates(Stheta: npt.ArrayLike) -> np.ndarray: ...


def screw_parameters_from_screw_axis(
        screw_axis: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray, float]: ...


def screw_axis_from_screw_parameters(
        q: npt.ArrayLike, s_axis: npt.ArrayLike, h: float) -> np.ndarray: ...


def screw_axis_from_exponential_coordinates(
        Stheta: npt.ArrayLike) -> Tuple[np.ndarray, float]: ...


def screw_axis_from_screw_matrix(screw_matrix: npt.ArrayLike) -> np.ndarray: ...


def exponential_coordinates_from_screw_axis(
        screw_axis: npt.ArrayLike, theta: float) -> np.ndarray: ...


def exponential_coordinates_from_transform_log(
        transform_log: npt.ArrayLike, check: bool = ...) -> np.ndarray: ...


def screw_matrix_from_screw_axis(screw_axis: npt.ArrayLike) -> np.ndarray: ...


def screw_matrix_from_transform_log(
        transform_log: npt.ArrayLike) -> np.ndarray: ...


def transform_log_from_exponential_coordinates(
        Stheta: npt.ArrayLike) -> np.ndarray: ...


def transform_log_from_screw_matrix(
        screw_matrix: npt.ArrayLike, theta: float) -> np.ndarray: ...


def transform_from_exponential_coordinates(
        Stheta: npt.ArrayLike, check: bool = ...) -> np.ndarray: ...


def transform_from_transform_log(
        transform_log: npt.ArrayLike) -> np.ndarray: ...
