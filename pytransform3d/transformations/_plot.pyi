import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union


def plot_transform(
        ax: Union[None, Axes3D] = ..., A2B: Union[None, npt.ArrayLike] = ...,
        s: float = ..., ax_s: float = ..., name: Union[None, str] = ...,
        strict_check: bool = ..., **kwargs) -> Axes3D: ...


def plot_screw(
        ax: Union[None, Axes3D] = ..., q: npt.ArrayLike = ...,
        s_axis: npt.ArrayLike = ..., h: float = ..., theta: float = ...,
        A2B: Union[None, npt.ArrayLike] = ..., s: float = ...,
        ax_s: float = ..., alpha: float = ..., **kwargs) -> Axes3D: ...
