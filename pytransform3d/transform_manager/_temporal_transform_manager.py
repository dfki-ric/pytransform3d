import abc
from typing import Any

import numpy as np

from ._transform_graph_base import TransformGraphBase
from ..transformations import (check_transform, invert_transform, concat)


class TimeTransform(abc.ABC):
    """Base class passed to TemporalTransformManager.add_transform() to deal with time.
    """
    
    @abc.abstractmethod
    def as_matrix(self, time) -> np.ndarray:
        ...


class TemporalTransformManager(TransformGraphBase):

    def __init__(self, strict_check=True, check=True):
        self._transforms = {}
        self._current_time = 0.0
        super(TemporalTransformManager, self).__init__(strict_check, check)
    
    @property
    def current_time(self):
        return self._current_time

    def set_time(self, time):
        self._current_time = time
    
    @property
    def transforms(self):
        """Rigid transformations between nodes."""
        return {tf_direction: transform.as_matrix(self.current_time) for
                tf_direction, transform in self._transforms.items()}

    def _transform_available(self, key):
        return key in self._transforms

    def _set_transform(self, key, A2B):
        self._transforms[key] = A2B

    def _get_transform(self, key):
        return self.transforms[key]

    def _del_transform(self, key):
        del self._transforms[key]

