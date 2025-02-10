import numpy as np
import numpy.typing as npt


def transform_requires_renormalization(
        A2B: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def check_transform(
        A2B: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def transform_from(R: npt.ArrayLike, p: npt.ArrayLike,
                   strict_check: bool = ...) -> np.ndarray: ...


def translate_transform(
        A2B: npt.ArrayLike, p: npt.ArrayLike, strict_check: bool = ...,
        check: bool = ...) -> np.ndarray: ...


def rotate_transform(
        A2B: npt.ArrayLike, R: npt.ArrayLike, strict_check: bool = ...,
        check: bool = ...) -> np.ndarray: ...


def pq_from_transform(
        A2B: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def transform_log_from_transform(
        A2B: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...


def exponential_coordinates_from_transform(
        A2B: npt.ArrayLike, strict_check: bool = ...,
        check: bool = ...) -> np.ndarray: ...
