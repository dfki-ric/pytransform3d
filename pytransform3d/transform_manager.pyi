import abc

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, List, Union, Set, Hashable, Any


class TransformGraphBase(abc.ABC):
    strict_check: bool
    check: bool
    nodes: List[Hashable]
    i: List[int]
    j: List[int]
    transform_to_ij_index = Dict[Tuple[Hashable, Hashable], int]
    connections: sp.csr_matrix
    dist: np.ndarray
    predecessors: np.ndarray
    _cached_shortest_paths: Dict[Tuple[int, int], List[Hashable]]

    def __init__(self, strict_check: bool = ...,
                 check: bool = ...) -> "TransformGraphBase": ...

    @property
    @abc.abstractmethod
    def transforms(self) -> Dict[Tuple[Hashable, Hashable], np.ndarray]: ...

    def has_frame(self, frame: Hashable) -> bool: ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: Any) -> "TransformGraphBase": ...

    def _check_transform(self, A2B: Any) -> Any: ...

    def _recompute_shortest_path(self): ...

    def remove_transform(
            self, from_frame: Hashable,
            to_frame: Hashable) -> "TransformGraphBase": ...

    def get_transform(
            self, from_frame: Hashable, to_frame: Hashable) -> Any: ...

    @abc.abstractmethod
    def _invert_transform(self, A2B: Any) -> Any: ...

    def _shortest_path(self, i: int, j: int) -> List[Hashable]: ...

    @abc.abstractmethod
    def _path_transform(self, path: List[Hashable]) -> Any: ...

    def connected_components(self) -> int: ...

    def check_consistency(self) -> bool: ...


class TransformManager(TransformGraphBase):
    _transforms: Dict[Tuple[Hashable, Hashable], np.ndarray]

    def __init__(self, strict_check: bool = ...,
                 check: bool = ...) -> "TransformManager": ...

    @property
    def transforms(self) -> Dict[Tuple[Hashable, Hashable], np.ndarray]: ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: np.ndarray) -> "TransformManager": ...

    def _check_transform(self, A2B: npt.ArrayLike) -> np.ndarray: ...

    def get_transform(self, from_frame: Hashable,
                      to_frame: Hashable) -> np.ndarray: ...

    def _invert_transform(self, A2B: np.ndarray) -> np.ndarray: ...

    def _path_transform(self, path: List[Hashable]) -> np.ndarray: ...

    def plot_frames_in(
            self, frame: Hashable, ax: Union[None, Axes3D] = ...,
            s: float = ..., ax_s: float = ..., show_name: bool = ...,
            whitelist: Union[List[str], None] = ..., **kwargs) -> Axes3D: ...

    def plot_connections_in(
            self, frame: Hashable, ax: Union[None, Axes3D] = ...,
            ax_s: float = ...,
            whitelist: Union[List[Hashable], None] = ...,
            **kwargs) -> Axes3D: ...

    def _whitelisted_nodes(
            self, whitelist: Union[None, List[Hashable]]) -> Set[Hashable]: ...

    def write_png(self, filename: str, prog: Union[str, None] = ...): ...

    def to_dict(self) -> Dict[str, Any]: ...

    @staticmethod
    def from_dict(tm_dict: Dict[str, Any]) -> "TransformManager": ...

    def set_transform_manager_state(self, tm_dict: Dict[str, Any]): ...
