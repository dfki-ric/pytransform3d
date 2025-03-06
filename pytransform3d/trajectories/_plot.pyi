import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union

def plot_trajectory(
    ax: Union[None, Axes3D] = ...,
    P: Union[None, npt.ArrayLike] = ...,
    normalize_quaternions: bool = ...,
    show_direction: bool = ...,
    n_frames: int = ...,
    s: float = ...,
    ax_s: float = ...,
    **kwargs,
) -> Axes3D: ...
