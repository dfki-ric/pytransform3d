import numpy as np
import numpy.typing as npt


def transform_requires_renormalization(
        A2B: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def check_transform(
        A2B: npt.ArrayLike, strict_check: bool = ...) -> np.ndarray: ...
