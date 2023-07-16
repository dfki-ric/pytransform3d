from typing import Dict, Tuple, List, Union, Set, Hashable, Any
import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
<<<<<<< HEAD:pytransform3d/transform_manager/_transform_manager.pyi
from ._transform_graph_base import TransformGraphBase


PYDOT_AVAILABLE: bool
=======
from typing import Dict, Tuple, List, Union, Set, Hashable, Any, Protocol, runtime_checkable


class TransformBase(abc.ABC):
    
    @abc.abstractmethod
    def get_matrix(self, time: float):
        ...

    @abc.abstractmethod
    def invert(self) -> "TransformBase":
        """Implements invert on a generic transformation"""
        ... 

    @abc.abstractmethod
    def concat(self, other: TransformBase) -> "TransformBase":
        """Implements concat on a transform with potentially different time base"""
        ...


class TransformGraphBase(abc.ABC):
    strict_check: bool
    check: bool
    nodes: List[Hashable]
    transforms: Dict[Tuple[Hashable, Hashable], TransformBase]
    i: List[int]
    j: List[int]
    transform_to_ij_index = Dict[Tuple[Hashable, Hashable], int]
    connections: sp.csr_matrix
    dist: np.ndarray
    predecessors: np.ndarray
    _cached_shortest_paths: Dict[Tuple[int, int], List[Hashable]]

    def has_frame(self, frame: Hashable) -> bool: ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: Any) -> "TransformGraphBase": ...

    def _check_transform(self, A2B: Any) -> Any: ...

    def _recompute_shortest_path(self): ...

    def remove_transform(
            self, from_frame: Hashable,
            to_frame: Hashable) -> "TransformManager": ...

    def get_transform(
            self, from_frame: Hashable, to_frame: Hashable) -> Any: ...

    @abc.abstractmethod
    def _invert_transform(self, A2B: Any) -> Any: ...

    def _shortest_path(self, i: int, j: int) -> List[Hashable]: ...

    @abc.abstractmethod
    def _path_transform(self, path: List[Hashable]) -> Any: ...

    def connected_components(self) -> int: ...

    def check_consistency(self) -> bool: ...
>>>>>>> 2665b8df (try to implement TemporalTranformManager):pytransform3d/transform_manager.pyi


class TransformManager(TransformGraphBase):
    _transforms: Dict[Tuple[Hashable, Hashable], np.ndarray]

    def __init__(self, strict_check: bool = ...,
                 check: bool = ...) -> "TransformManager": ...

    @property
    def transforms(self) -> Dict[Tuple[Hashable, Hashable], np.ndarray]: ...

    def _check_transform(self, A2B: npt.ArrayLike) -> np.ndarray: ...

    def _invert_transform(self, A2B: np.ndarray) -> np.ndarray: ...

    def _path_transform(self, path: List[Hashable]) -> np.ndarray: ...

    def _transform_available(self, key: Tuple[Hashable, Hashable]) -> bool: ...

    def _set_transform(self, key: Tuple[Hashable, Hashable], A2B: Any): ...

    def _get_transform(self, key: Tuple[Hashable, Hashable]) -> Any: ...

    def _del_transform(self, key: Tuple[Hashable, Hashable]): ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: np.ndarray) -> "TransformManager": ...

    def get_transform(self, from_frame: Hashable,
                      to_frame: Hashable) -> np.ndarray: ...

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
