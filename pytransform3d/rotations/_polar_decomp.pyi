import numpy as np
import numpy.typing as npt


def robust_polar_decomposition(
        A: npt.ArrayLike, n_iter: int = ...,
        eps: float = ...) -> np.ndarray: ...
