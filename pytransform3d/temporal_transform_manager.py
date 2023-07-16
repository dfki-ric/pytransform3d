import abc
from typing import Any

import numpy as np

from .transform_manager import TransformGraphBase


def NumpyTransform():
    
    def __init__(self, A2B):
        self._A2B = A2B

    def invert(self) -> "NumpyTransform":
        return np.linalg.inv(self._A2B)
    
    def get_matrix(self, time) -> "NumpyTransform":
        return self._A2B
    
    def concat(self, other) -> "NumpyTransform":
        return other.get_matrix().dot(self._A2B)


class TemporalTransformManager(TransformGraphBase):

    def __init__(self):
        super().__init__()
    
    def _check_transform(self, A2B: Any):
        return A2B
    
    def _invert_transform(self, A2B):
        """Invert rigid transformation stored in the tree."""
        return A2B.invert()
    
    def _path_transform(self, path):
        A2B = NumpyTransform(np.eye(4))
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B = A2B.concat(self.get_transform(from_f, to_f))
        return A2B
