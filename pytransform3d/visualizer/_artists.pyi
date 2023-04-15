import typing

import numpy as np
import numpy.typing as npt
import open3d as o3d
from typing import List, Union, Any, Dict
from ..transform_manager import TransformManager
if typing.TYPE_CHECKING:
    from ._figure import Figure


class Artist:
    def add_artist(self, figure: Figure): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Line3D(Artist):
    def __init__(self, P: npt.ArrayLike, c: npt.ArrayLike = ...): ...

    def set_data(self, P: npt.ArrayLike,
                 c: Union[None, npt.ArrayLike] = ...): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class PointCollection3D(Artist):
    def __init__(self, P: npt.ArrayLike, s: float = ...,
                 c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, P: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Vector3D(Artist):
    def __init__(self, start: npt.ArrayLike, direction: npt.ArrayLike,
                 c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, start: npt.ArrayLike, direction: npt.ArrayLike,
                 c: npt.ArrayLike = ...): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Frame(Artist):
    def __init__(self, A2B: npt.ArrayLike,
                 label: Union[None, str] = ..., s: float = ...): ...

    def set_data(self, A2B: npt.ArrayLike, label: Union[None, str] = ...): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Trajectory(Artist):
    def __init__(self, H: npt.ArrayLike, n_frames: int = ..., s: float = ...,
                 c: npt.ArrayLike = ...): ...

    def set_data(self, H: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Sphere(Artist):
    def __init__(self, radius: float = ..., A2B: npt.ArrayLike = ...,
                 resolution: int = ..., c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Box(Artist):
    def __init__(self, size: npt.ArrayLike = ..., A2B: npt.ArrayLike = ...,
                 c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Cylinder(Artist):
    def __init__(self, length: float = ..., radius: float = ...,
                 A2B: npt.ArrayLike = ..., resolution: int = ...,
                 split: int = ..., c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Mesh(Artist):
    def __init__(self, filename: str, A2B: npt.ArrayLike = ...,
                 s: npt.ArrayLike = ..., c: Union[None, npt.ArrayLike] = ...,
                 convex_hull: bool = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Ellipsoid(Artist):
    def __init__(
            self, radii: npt.ArrayLike, A2B: npt.ArrayLike = ...,
            resolution: int = ..., c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Capsule(Artist):
    def __init__(
            self, height: float = ..., radius: float = ...,
            A2B: npt.ArrayLike = ..., resolution: int = ...,
            c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Cone(Artist):
    def __init__(
            self, height: float = ..., radius: float = ...,
            A2B: npt.ArrayLike = ..., resolution: int = ...,
            c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(self, A2B: npt.ArrayLike): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Plane(Artist):
    def __init__(
            self, normal: npt.ArrayLike = ..., d: Union[None, float] = ...,
            point_in_plane: Union[None, npt.ArrayLike] = ...,
            s: float = ..., c: Union[None, npt.ArrayLike] = ...): ...

    def set_data(
            self, normal: npt.ArrayLike, d: Union[None, float] = ...,
            point_in_plane: Union[None, npt.ArrayLike] = ...,
            s: float = ..., c: Union[None, npt.ArrayLike] = ...): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Camera(Artist):
    def __init__(
            self, M: npt.ArrayLike, cam2world: Union[None, npt.ArrayLike] = ...,
            virtual_image_distance: float = ...,
            sensor_size: npt.ArrayLike = ..., strict_check: bool = ...): ...

    def set_data(
            self, M: Union[None, npt.ArrayLike],
            cam2world: Union[None, npt.ArrayLike],
            virtual_image_distance: Union[None, float],
            sensor_size: Union[None, npt.ArrayLike]): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


class Graph(Artist):
    def __init__(
            self, tm: TransformManager, frame: str, show_frames: bool = ...,
            show_connections: bool = ..., show_visuals: bool = ...,
            show_collision_objects: bool = ..., show_name: bool = ...,
            whitelist: Union[None, List[str]] = ...,
            convex_hull_of_collision_objects: bool = ..., s: float = ...): ...

    def set_data(self): ...

    @property
    def geometries(self) -> List[o3d.geometry.Geometry3D]: ...


def _objects_to_artists(objects: List[Any]) -> Dict[str, Artist]: ...
