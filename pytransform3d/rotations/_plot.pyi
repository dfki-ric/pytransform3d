import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union


def plot_basis(
        ax: Union[None, Axes3D] = ..., R: Union[None, npt.ArrayLike] = ...,
        p: npt.ArrayLike = ..., s: float = ..., ax_s: float = ...,
        strict_check: bool = ..., **kwargs) -> Axes3D: ...


def plot_axis_angle(
        ax: Union[None, Axes3D] = ..., a: npt.ArrayLike = ...,
        p: npt.ArrayLike = ..., s: float = ..., ax_s: float = ...,
        **kwargs) -> Axes3D: ...


def plot_bivector(
        ax: Union[None, Axes3D] = ..., a: Union[None, npt.ArrayLike] = ...,
        b: Union[None, npt.ArrayLike] = ..., ax_s: float = ...): ...
