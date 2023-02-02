import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, List, Union, Set, Hashable, Any


class TransformManager(object):
    strict_check: bool
    check: bool
    transforms: Dict[Tuple[Hashable, Hashable], np.ndarray]
    nodes: List[Hashable]
    i: List[int]
    j: List[int]
    transform_to_ij_index = Dict[Tuple[Hashable, Hashable], int]
    connections: sp.csr_matrix
    dist: np.ndarray
    predecessors: np.ndarray
    _cached_shortest_paths: Dict[Tuple[int, int], List[Hashable]]

    def __init__(self, strict_check: bool = ..., check: bool = ...): ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: np.ndarray) -> "TransformManager": ...

    def remove_transform(
            self, from_frame: Hashable,
            to_frame: Hashable) -> "TransformManager": ...

    def has_frame(self, frame: Hashable) -> bool: ...

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

    def check_consistency(self) -> bool: ...

    def connected_components(self) -> int: ...

    def write_png(self, filename: str, prog: Union[str, None] = ...): ...

    def _recompute_shortest_path(self): ...

    def _shortest_path(self, i: int, j: int) -> List[str]: ...

    def _path_transform(self, path: List[Hashable]) -> np.ndarray: ...

    def to_dict(self) -> Dict[Any, Any]: ...

    @staticmethod
    def from_dict(tm_dict: Dict[Any, Any]) -> "TransformManager": ...
