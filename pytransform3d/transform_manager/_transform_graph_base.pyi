import abc
from typing import Dict, Tuple, List, Hashable, Any
import scipy.sparse as sp
import numpy as np
import numpy.typing as npt


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

    @abc.abstractmethod
    def _check_transform(self, A2B: Any) -> Any: ...

    @abc.abstractmethod
    def _invert_transform(self, A2B: Any) -> Any: ...

    @abc.abstractmethod
    def _path_transform(self, path: List[Hashable]) -> Any: ...

    def has_frame(self, frame: Hashable) -> bool: ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: Any) -> "TransformGraphBase": ...

    def _recompute_shortest_path(self): ...

    def remove_transform(
            self, from_frame: Hashable,
            to_frame: Hashable) -> "TransformGraphBase": ...

    def get_transform(
            self, from_frame: Hashable, to_frame: Hashable) -> Any: ...

    def _shortest_path(self, i: int, j: int) -> List[Hashable]: ...

    def connected_components(self) -> int: ...

    def check_consistency(self) -> bool: ...
