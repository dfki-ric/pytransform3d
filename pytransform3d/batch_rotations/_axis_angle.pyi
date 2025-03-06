import numpy as np
import numpy.typing as npt
from typing import Union

def norm_axis_angles(a: npt.ArrayLike) -> np.ndarray: ...
def matrices_from_compact_axis_angles(
    A: Union[npt.ArrayLike, None] = ...,
    axes: Union[np.ndarray, None] = ...,
    angles: Union[np.ndarray, None] = ...,
    out: Union[np.ndarray, None] = ...,
) -> np.ndarray: ...
