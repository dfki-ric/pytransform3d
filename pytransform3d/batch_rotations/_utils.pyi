import numpy as np
import numpy.typing as npt
from typing import Union

def norm_vectors(
    V: npt.ArrayLike, out: Union[npt.ArrayLike, None] = ...
) -> np.ndarray: ...
def angles_between_vectors(
    A: npt.ArrayLike, B: npt.ArrayLike
) -> np.ndarray: ...
def cross_product_matrices(V: npt.ArrayLike) -> np.ndarray: ...
