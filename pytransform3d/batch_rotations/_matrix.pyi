import numpy as np
import numpy.typing as npt
from typing import Union

def axis_angles_from_matrices(
    Rs: npt.ArrayLike,
    traces: Union[np.ndarray, None] = ...,
    out: Union[np.ndarray, None] = ...,
) -> np.ndarray: ...
def quaternions_from_matrices(
    Rs: npt.ArrayLike, out: Union[np.ndarray, None] = ...
) -> np.ndarray: ...
