import numpy as np
import numpy.typing as npt


def mrp_near_singularity(mrp: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def concatenate_mrp(mrp1: npt.ArrayLike, mrp2: npt.ArrayLike) -> np.ndarray: ...
