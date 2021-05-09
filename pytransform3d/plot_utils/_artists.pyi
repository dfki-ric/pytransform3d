import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib import artist
from matplotlib.patches import FancyArrowPatch
from matplotlib.backend_bases import RendererBase
from typing import Union


class Frame(artist.Artist):
    def __init__(self, A2B: npt.ArrayLike, label: Union[str, None] = ...,
                 s: float = ..., **kwargs): ...

    def set_data(self, A2B: npt.ArrayLike, label: Union[str, None] = ...): ...

    @artist.allow_rasterization
    def draw(self, renderer: RendererBase, *args, **kwargs): ...

    def add_frame(self, axis: Axes3D): ...


class LabeledFrame(Frame):
    def __init__(self, A2B: npt.ArrayLike, label: Union[str, None] = ...,
                 s: float = ..., **kwargs): ...

    def set_data(self, A2B: npt.ArrayLike, label=None): ...

    @artist.allow_rasterization
    def draw(self, renderer: RendererBase, *args, **kwargs): ...

    def add_frame(self, axis: Axes3D): ...


class Trajectory(artist.Artist):
    trajectory : Line3D

    def __init__(self, H: npt.ArrayLike, show_direction: bool = ...,
                 n_frames: int = ..., s: float = ..., **kwargs): ...

    def set_data(self, H: npt.ArrayLike): ...

    @artist.allow_rasterization
    def draw(self, renderer: RendererBase, *args, **kwargs): ...

    def add_trajectory(self, axis: Axes3D): ...


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs: npt.ArrayLike, ys: npt.ArrayLike, zs: npt.ArrayLike,
                 *args, **kwargs): ...

    def set_data(self, xs: npt.ArrayLike, ys: npt.ArrayLike, zs: npt.ArrayLike): ...

    def draw(self, renderer: RendererBase): ...
