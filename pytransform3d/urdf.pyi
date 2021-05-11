import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from bs4.element import Tag
from transform_manager import TransformManager
from typing import Dict, Tuple, List, Any, Union, Type


class UrdfTransformManager(TransformManager):
    _joints: Dict[str, Tuple[str, str, np.ndarray, np.ndarray,
                             Tuple[float, float], str]]
    collision_objects: List[Any]
    visuals: List[Any]
    mesh_path: Union[None, str]
    package_dir: Union[None, str]

    def __init__(self, strict_check: bool = ..., check: bool = ...): ...

    def add_joint(
            self, joint_name: str, from_frame: str, to_frame: str,
            child2parent: npt.ArrayLike, axis: npt.ArrayLike,
            limits: Tuple[float, float] = ...,
            joint_type: str = ...): ...

    def set_joint(self, joint_name: str, value: float): ...

    def get_joint_limits(self, joint_name: str) -> Tuple[float, float]: ...

    def load_urdf(self, urdf_xml: str, mesh_path: Union[None, str] = ...,
                  package_dir: Union[None, str] = ...): ...

    def _add_joints(self, joints): ...

    def _parse_link(self, link, materials): ...

    def _parse_link_children(self, link, child_type, materials): ...

    def _parse_geometry(self, child, name, color): ...

    def plot_visuals(
            self, frame: str, ax: Union[None, Axes3D] = ..., ax_s: float = ...,
            wireframe: bool = ..., convex_hull_of_mesh: bool = ...,
            alpha: float = ...): ...

    def plot_collision_objects(
            self, frame: str, ax: Union[None, Axes3D] = ..., ax_s: float = ...,
            wireframe: bool = ..., convex_hull_of_mesh: bool = ...,
            alpha: float = ...): ...

    def _plot_objects(
            self, objects, frame: str, ax: Union[None, Axes3D] = ...,
            ax_s: float = ..., wireframe: bool = ...,
            convex_hull_of_mesh: bool = ..., alpha: float = ...): ...


class Geometry(object):
    frame: str
    mesh_path: Union[None, str]
    package_dir: Union[None, str]
    color: npt.ArrayLike

    def __init__(self, frame: str, mesh_path: Union[None, str],
                 package_dir: Union[None, str], color: str) -> "Geometry": ...

    def parse(self, xml: Tag): ...

    def plot(self, tm: UrdfTransformManager, frame: str,
             ax: Union[None, Axes3D] = ..., alpha: float = ...,
             wireframe: bool = ..., convex_hull: bool = ...): ...


class Box(Geometry): ...


class Sphere(Geometry): ...


class Cylinder(Geometry): ...


class Mesh(Geometry): ...


shape_classes: Dict[str, Union[Type[Geometry]]]
