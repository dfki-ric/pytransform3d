import numpy as np
import numpy.typing as npt
import open3d as o3d
from typing import Callable, Tuple, Any, Union, List
from ..transform_manager import TransformManager
from ._artists import (Artist, Frame, Trajectory, Sphere, Box, Cylinder, Mesh,
                       Ellipsoid, Capsule, Cone, Plane, Graph, Camera)


class Figure:
    def __init__(self, window_name: str = ..., width: int = ...,
                 height: int = ..., with_key_callbacks: bool = ...): ...

    def add_geometry(self, geometry: o3d.geometry.Geometry3D): ...

    def _remove_geometry(self, geometry: o3d.geometry.Geometry3D): ...

    def update_geometry(self, geometry: o3d.geometry.Geometry3D): ...

    def remove_artist(self, artist: Artist): ...

    def set_line_width(self, line_width: float): ...

    def set_zoom(self, zoom: float): ...

    def animate(self, callback: Callable, n_frames: int,
                loop: bool = ..., fargs: Tuple[Any, ...] = ...): ...

    def view_init(self, azim: float = ..., elev: float = ...): ...

    def plot(self, P: npt.ArrayLike, c: npt.ArrayLike = ...): ...

    def scatter(self, P: npt.ArrayLike, s: float = ...,
                c: Union[None, npt.ArrayLike] = ...): ...

    def plot_vector(self, start: npt.ArrayLike, direction: npt.ArrayLike,
                    c: npt.ArrayLike = ...): ...

    def plot_basis(
            self, R: Union[None, npt.ArrayLike] = ..., p: npt.ArrayLike = ...,
            s: float = ..., strict_check: bool = ...) -> Frame: ...

    def plot_transform(
            self, A2B: Union[None, npt.ArrayLike] = ..., s: float = ...,
            name: Union[str, None] = ..., strict_check: bool = ...) -> Frame: ...

    def plot_trajectory(
            self, P: npt.ArrayLike, n_frames: int = ..., s: float = ...,
            c: npt.ArrayLike = ...) -> Trajectory: ...

    def plot_sphere(
            self, radius: float = ..., A2B: npt.ArrayLike = ...,
            resolution: int = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Sphere: ...

    def plot_box(
            self, size: npt.ArrayLike = ..., A2B: npt.ArrayLike = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Box: ...

    def plot_cylinder(
            self, length: float = ..., radius: float = ...,
            A2B: npt.ArrayLike = ..., resolution: int = ...,
            split: int = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Cylinder: ...

    def plot_mesh(
            self, filename: str, A2B: npt.ArrayLike = ...,
            s: npt.ArrayLike = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Mesh: ...

    def plot_ellipsoid(
            self, radii: npt.ArrayLike = ..., A2B: npt.ArrayLike = ...,
            resolution: int = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Ellipsoid: ...

    def plot_capsule(
            self, height: float = ..., radius: float = ...,
            A2B: npt.ArrayLike = ..., resolution: int = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Capsule: ...

    def plot_cone(
            self, height: float = ..., radius: float = ...,
            A2B: npt.ArrayLike = ..., resolution: int = ...,
            c: Union[None, npt.ArrayLike] = ...) -> Cone: ...

    def plot_plane(
            self, normal: npt.ArrayLike = ..., d: Union[None, float] = ...,
            point_in_plane: Union[None, npt.ArrayLike] = ...,
            s: float = ..., c: Union[None, npt.ArrayLike] = ...) -> Plane: ...

    def plot_graph(
            self, tm: TransformManager, frame: str, show_frames: bool = ...,
            show_connections: bool = ..., show_visuals: bool = ...,
            show_collision_objects: bool = ..., show_name: bool = ...,
            whitelist: Union[None, List[str]] = ...,
            convex_hull_of_collision_objects: bool = ...,
            s: float = ...) -> Graph: ...

    def plot_camera(
            self, M: npt.ArrayLike,
            cam2world: Union[None, npt.ArrayLike] = ...,
            virtual_image_distance: float = ...,
            sensor_size: npt.ArrayLike = ...,
            strict_check: bool = ...) -> Camera: ...

    def save_image(self, filename: str): ...

    def show(self): ...


def figure(window_name: str = ..., width: int = ..., height: int = ...,
           with_key_callbacks: bool = ...) -> Figure: ...
