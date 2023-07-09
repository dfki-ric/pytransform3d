import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TransformBase(abc.ABC):
    """Represents the transformation from 'from_frame' to 'to_frame'
    """

    @abc.abstractmethod
    def as_homogenous_matrix(self, time: Optional[float] = None) -> np.ndarray:
        """
        Returns
        -------
        A2B : array, shape (4, 4)
            Validated transform from frame A to frame B
        """
        ...


@dataclass
class TransformNumpy(TransformBase):

    from_frame: str
    to_frame: str
    A2B: np.ndarray

    def as_homogenous_matrix(self, time: Optional[float] = None) -> np.ndarray:
        return self.A2B
