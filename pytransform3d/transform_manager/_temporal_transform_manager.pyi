import abc
from typing import Dict, Tuple, Hashable, Any

import numpy as np
import numpy.typing as npt

from ._transform_graph_base import TransformGraphBase


class TimeVaryingTransform(abc.ABC):

    @abc.abstractmethod
    def as_matrix(self, query_time: float) -> np.ndarray: ...

    @abc.abstractmethod
    def check_transforms(self) -> "TimeVaryingTransform": ...


class StaticTransform(TimeVaryingTransform):
    _A2B: npt.ArrayLike

    def __init__(self, A2B: npt.ArrayLike): ...

    def as_matrix(self, query_time: float) -> np.ndarray: ...

    def check_transforms(self) -> "StaticTransform": ...


class NumpyTimeseriesTransform(TimeVaryingTransform):
    time: np.ndarray
    _pqs: np.ndarray

    def __init__(self, time: npt.ArrayLike, pqs: npt.ArrayLike): ...

    def as_matrix(self, query_time: float) -> np.ndarray: ...

    def check_transforms(self) -> "NumpyTimeseriesTransform": ...

    def _interpolate_pq_using_sclerp(self, query_time) -> np.ndarray: ...


class TemporalTransformManager(TransformGraphBase):
    _transforms: Dict[Tuple[Hashable, Hashable], TimeVaryingTransform]
    _current_time: float

    def __init__(self, strict_check: bool = ..., check: bool = ...): ...

    @property
    def transforms(self) -> Dict[Tuple[Hashable, Hashable], np.ndarray]: ...

    @property
    def current_time(self) -> float: ...

    @current_time.setter
    def current_time(self, query_time: float): ...

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
                              query_time: float) -> np.ndarray: ...
