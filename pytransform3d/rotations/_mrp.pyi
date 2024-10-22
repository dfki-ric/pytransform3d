import numpy as np
import numpy.typing as npt


def norm_mrp(mrp: npt.ArrayLike) -> np.ndarray: ...


def mrp_near_singularity(mrp: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def mrp_double(mrp: npt.ArrayLike) -> np.ndarray: ...


def concatenate_mrp(mrp1: npt.ArrayLike, mrp2: npt.ArrayLike) -> np.ndarray: ...
