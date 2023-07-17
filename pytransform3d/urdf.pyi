from typing import Dict, Tuple, List, Union, Type, Hashable
import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from lxml.etree import Element
from .transform_manager import TransformManager


class UrdfTransformManager(TransformManager):
    _joints: Dict[str, Tuple[Hashable, Hashable, np.ndarray, np.ndarray,
                             Tuple[float, float], str]]
    collision_objects: List[Geometry]
    visuals: List[Geometry]

    def __init__(self, strict_check: bool = ..., check: bool = ...): ...

    def add_joint(
            self, joint_name: str, from_frame: Hashable, to_frame: Hashable,
            child2parent: npt.ArrayLike, axis: npt.ArrayLike,
            limits: Tuple[float, float] = ...,
            joint_type: str = ...): ...

    def set_joint(self, joint_name: str, value: float): ...

    def get_joint_limits(self, joint_name: str) -> Tuple[float, float]: ...

    def load_urdf(self, urdf_xml: str, mesh_path: Union[None, str] = ...,
                  package_dir: Union[None, str] = ...): ...

    def plot_visuals(
            self, frame: Hashable, ax: Union[None, Axes3D] = ...,
            ax_s: float = ..., wireframe: bool = ...,
            convex_hull_of_mesh: bool = ..., alpha: float = ...): ...

    def plot_collision_objects(
            self, frame: Hashable, ax: Union[None, Axes3D] = ...,
            ax_s: float = ..., wireframe: bool = ...,
            convex_hull_of_mesh: bool = ..., alpha: float = ...): ...

    def _plot_objects(
            self, objects, frame: Hashable, ax: Union[None, Axes3D] = ...,
            ax_s: float = ..., wireframe: bool = ...,
            convex_hull_of_mesh: bool = ..., alpha: float = ...): ...


class Link(object):
    name: Union[None, str]
    visuals: List[Geometry]
    collision_objects: List[Geometry]
    transforms: List[Tuple[str, str, np.ndarray]]
    inertial_frame: np.ndarray
    mass: float
    inertia: np.ndarray


class Joint(object):
    child: Union[None, str]
    parent: Union[None, str]
    child2parent: np.ndarray
    joint_name: Union[None, str]
    joint_axis: Union[None, np.ndarray]
    joint_type: str
    limits: Tuple[float, float]


class Geometry(object):
    frame: str
    mesh_path: Union[None, str]
    package_dir: Union[None, str]
    color: npt.ArrayLike

    def __init__(self, frame: str, mesh_path: Union[None, str],
                 package_dir: Union[None, str], color: str): ...

    def parse(self, xml: Element): ...

    def plot(self, tm: UrdfTransformManager, frame: Hashable,
             ax: Union[None, Axes3D] = ..., alpha: float = ...,
             wireframe: bool = ..., convex_hull: bool = ...): ...


class Box(Geometry):
    size: np.ndarray


class Sphere(Geometry):
    radius: float


class Cylinder(Geometry):
    radius: float
    length: float


class Mesh(Geometry):
    filename: Union[None, str]
    scale: np.ndarray


class UrdfException(Exception): ...


def parse_urdf(
        urdf_xml: str, mesh_path: Union[str, None] = ...,
        package_dir: Union[str, None] = ...,
        strict_check: bool = ...) -> Tuple[str, List[Link], List[Joint]]: ...


def initialize_urdf_transform_manager(
        tm : UrdfTransformManager, robot_name: str, links: List[Link],
        joints: List[Joint]): ...


shape_classes: Dict[str, Union[Type[Geometry]]]
