import numpy as np
import numpy.typing as npt
from typing import Tuple, Callable

def frechet_mean(
    samples: npt.ArrayLike,
    mean0: npt.ArrayLike,
    exp: Callable,
    log: Callable,
    inv: Callable,
    concat_one_to_one: Callable,
    concat_many_to_one: Callable,
    n_iter: int = ...,
) -> Tuple[np.ndarray, np.ndarray]: ...
def estimate_gaussian_rotation_matrix_from_samples(
    samples: npt.ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]: ...
def estimate_gaussian_transform_from_samples(
    samples: npt.ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]: ...
