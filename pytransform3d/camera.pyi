import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union


def make_world_grid(
        n_lines: int = ..., n_points_per_line: int = ...,
        xlim: npt.ArrayLike = ..., ylim: npt.ArrayLike = ...) -> np.ndarray: ...


def make_world_line(
        p1: npt.ArrayLike, p2: npt.ArrayLike, n_points: int) -> np.ndarray: ...


def cam2sensor(P_cam: npt.ArrayLike, focal_length: float,
               kappa: float = ...) -> np.ndarray: ...


def sensor2img(P_sensor: npt.ArrayLike, sensor_size: npt.ArrayLike,
               image_size: npt.ArrayLike,
               image_center: Union[npt.ArrayLike, None] = ...) -> np.ndarray: ...


def world2image(P_world: npt.ArrayLike, cam2world: npt.ArrayLike,
                sensor_size: npt.ArrayLike, image_size: npt.ArrayLike,
                focal_length: float, image_center: Union[npt.ArrayLike, None] = ...,
                kappa: float = ...) -> np.ndarray: ...


def plot_camera(
        ax: Union[None, Axes3D] = ..., M: Union[None, npt.ArrayLike] = ...,
        cam2world: Union[None, npt.ArrayLike] = ...,
        virtual_image_distance: float = ..., sensor_size: npt.ArrayLike = ...,
        ax_s: float = ..., strict_check: bool = ..., **kwargs) -> Axes3D: ...
