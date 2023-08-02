import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union


def plot_box(ax: Union[None, Axes3D] = ..., size: npt.ArrayLike = ...,
             A2B: npt.ArrayLike = ..., ax_s: float = ...,
             wireframe: bool = ..., color: str = ..., alpha: float = ...) -> Axes3D: ...


def plot_sphere(
        ax: Union[None, Axes3D] = ..., radius: float = ...,
        p: npt.ArrayLike = ..., ax_s: float = ..., wireframe: bool = ...,
        n_steps: int = ..., alpha: float = ...,
        color: str = ...) -> Axes3D: ...


def plot_spheres(
        ax: Union[None, Axes3D] = ..., radius: npt.ArrayLike = ...,
        p: npt.ArrayLike = ..., ax_s: float = ..., wireframe: bool = ...,
        n_steps: int = ..., alpha: npt.ArrayLike = ...,
        color: npt.ArrayLike = ...) -> Axes3D: ...


def plot_cylinder(
        ax: Union[None, Axes3D] = ..., length: float = ...,
        radius: float = ..., thickness: float = ..., A2B: npt.ArrayLike = ...,
        ax_s: float = ..., wireframe: bool = ..., n_steps: int = ...,
        alpha: float = ..., color: str = ...) -> Axes3D: ...


def plot_mesh(
        ax: Union[None, Axes3D] = ..., filename: Union[None, str] = ...,
        A2B: npt.ArrayLike = ..., s: npt.ArrayLike = ..., ax_s: float = ...,
        wireframe: bool = ..., convex_hull: bool = ..., alpha: float = ...,
        color: str = ...) -> Axes3D: ...


def plot_ellipsoid(
        ax: Union[None, Axes3D] = ..., radii: npt.ArrayLike = ...,
        A2B: npt.ArrayLike = ..., ax_s: float = ..., wireframe: bool = ...,
        n_steps: int = ..., alpha: float = ...,
        color: str = ...) -> Axes3D: ...


def plot_capsule(
        ax: Union[None, Axes3D] = ..., A2B: npt.ArrayLike = ...,
        height: float = ..., radius: float = ..., ax_s: float = ...,
        wireframe: bool = ..., n_steps: int = ..., alpha: float = ...,
        color: str = ...) -> Axes3D: ...


def plot_cone(
        ax: Union[None, Axes3D] = ..., height: float = ...,
        radius: float = ..., A2B: npt.ArrayLike = ...,
        ax_s: float = ..., wireframe: bool = ..., n_steps: int = ...,
        alpha: float = ..., color: str = ...) -> Axes3D: ...


def plot_vector(
        ax: Union[None, Axes3D] = ..., start: npt.ArrayLike = ...,
        direction: npt.ArrayLike = ..., s: float = ..., arrowstyle: str = ...,
        ax_s: float = ..., **kwargs) -> Axes3D: ...


def plot_length_variable(
        ax: Union[None, Axes3D] = ..., start: npt.ArrayLike = ...,
        end: npt.ArrayLike = ..., name: str = ..., above: bool = ...,
        ax_s: float = ..., color: str = ..., **kwargs) -> Axes3D: ...
