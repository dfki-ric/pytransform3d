import numpy as np
import numpy.typing as npt
from typing import Union

def active_matrices_from_intrinsic_euler_angles(
    basis1: int,
    basis2: int,
    basis3: int,
    e: npt.ArrayLike,
    out: Union[npt.ArrayLike, None] = ...,
) -> np.ndarray: ...
def active_matrices_from_extrinsic_euler_angles(
    basis1: int,
    basis2: int,
    basis3: int,
    e: npt.ArrayLike,
    out: Union[npt.ArrayLike, None] = ...,
) -> np.ndarray: ...
