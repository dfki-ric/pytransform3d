import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union

def plot_vector(
    ax: Union[None, Axes3D] = ...,
    start: npt.ArrayLike = ...,
    direction: npt.ArrayLike = ...,
    s: float = ...,
    arrowstyle: str = ...,
    ax_s: float = ...,
    **kwargs,
) -> Axes3D: ...
def plot_length_variable(
    ax: Union[None, Axes3D] = ...,
    start: npt.ArrayLike = ...,
    end: npt.ArrayLike = ...,
    name: str = ...,
    above: bool = ...,
    ax_s: float = ...,
    color: str = ...,
    **kwargs,
) -> Axes3D: ...
