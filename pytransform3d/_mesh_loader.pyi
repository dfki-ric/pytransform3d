import abc
from typing import Any

import numpy.typing as npt


class MeshBase(abc.ABC):
    filename: str

    def __init__(self, filename: str): ...

    @abc.abstractmethod
    def load(self) -> bool: ...

    @abc.abstractmethod
    def convex_hull(self): ...

    @abc.abstractmethod
    def get_open3d_mesh(self) -> Any: ...

    @property
    @abc.abstractmethod
    def vertices(self) -> npt.ArrayLike: ...

    @property
    @abc.abstractmethod
    def triangles(self) -> npt.ArrayLike: ...


def load_mesh(filename: str) -> MeshBase: ...
