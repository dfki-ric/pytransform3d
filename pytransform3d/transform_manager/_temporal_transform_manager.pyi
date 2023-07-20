import abc
from typing import Dict, Tuple, Hashable, Any

import numpy as np
import numpy.typing as npt

from ._transform_graph_base import TransformGraphBase


class TimeVaryingTransform(abc.ABC):

    @abc.abstractmethod
    def as_matrix(self, time: float) -> np.ndarray: ...

    @abc.abstractmethod
    def check_transforms(self) -> "TimeVaryingTransform": ...


class TemporalTransformManager(TransformGraphBase):
    _transforms: Dict[Tuple[Hashable, Hashable], TimeVaryingTransform]
    _current_time: float

    def __init__(self, strict_check: bool = ...,
                 check: bool = ...) -> "TemporalTransformManager": ...

    @property
    def transforms(self) -> Dict[Tuple[Hashable, Hashable], np.ndarray]: ...

    @property
    def current_time(self) -> float: ...

    @current_time.setter
    def current_time(self, time: float): ...

    def _check_transform(
            self, A2B: TimeVaryingTransform) -> TimeVaryingTransform: ...

    def _transform_available(self, key: Tuple[Hashable, Hashable]) -> bool: ...

    def _set_transform(self, key: Tuple[Hashable, Hashable],
                       A2B: TimeVaryingTransform): ...

    def _get_transform(self, key: Tuple[Hashable, Hashable]) -> Any: ...

    def _del_transform(self, key: Tuple[Hashable, Hashable]): ...

    def add_transform(self, from_frame: Hashable, to_frame: Hashable,
                      A2B: TimeVaryingTransform
                      ) -> "TemporalTransformManager": ...

    def get_transform(self, from_frame: Hashable,
                      to_frame: Hashable) -> np.ndarray: ...

    def get_transform_at_time(self, from_frame: Hashable,
                                to_frame: Hashable,
                                time: float) -> np.ndarray: ...
