from typing import Dict, Tuple, List, Union, Set, Hashable, Any

import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D

from ._transform_graph_base import TransformGraphBase


PYDOT_AVAILABLE: bool


class TransformManager(TransformGraphBase):
    _transforms: Dict[Tuple[Hashable, Hashable], np.ndarray]

    def __init__(self, strict_check: bool = ..., check: bool = ...): ...

    @property
    def transforms(self) -> Dict[Tuple[Hashable, Hashable], np.ndarray]: ...

    def _check_transform(self, A2B: npt.ArrayLike) -> np.ndarray: ...

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
