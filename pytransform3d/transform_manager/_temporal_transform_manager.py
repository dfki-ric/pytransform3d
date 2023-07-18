import abc
from typing import Dict, Tuple, Hashable

import numpy as np

from ._transform_graph_base import TransformGraphBase
from ..transformations import check_transform


class TimeVaryingTransform(abc.ABC):
    """Time-varying rigid transformation."""

    @abc.abstractmethod
    def as_matrix(self, time) -> np.ndarray:
        """Get transformation matrix at given time.

        Parameters
        ----------
        time : float
            Time

        Returns
        -------
        A2B_t : array, shape (4, 4)
            Homogeneous transformation matrix at given time.
        """

    @abc.abstractmethod
    def check_transforms(self) -> "TimeVaryingTransform":
        """Checks all transformations.

        Returns
        -------
        self : TimeVaryingTransform
            Validated transformations.
        """


class StaticTransform(TimeVaryingTransform):
    """Transformation, which does not change over time.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Homogeneous transformation matrix.
    """
    def __init__(self, A2B):
        self._A2B = A2B

    def as_matrix(self, time):
        return self._A2B
    
    def check_transforms(self):
        self._A2B = check_transform(self._A2B)
        return self


class TemporalTransformManager(TransformGraphBase):
    """Transform manager with time-varying transformations.

    Parameters
    ----------
    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrices are valid and requested nodes exist,
        which might significantly slow down some operations.
    """
    def __init__(self, strict_check=True, check=True):
        super(TemporalTransformManager, self).__init__(strict_check, check)
        self._transforms: Dict[Tuple[Hashable, Hashable], TimeVaryingTransform] = {}
        self._current_time: float = 0.0

    @property
    def current_time(self) -> float:
        """Current time at which we evaluate transformations."""
        return self._current_time

    def set_time(self, time: float):
        """Set current time at which we evaluate transformations."""
        self._current_time = time

    @property
    def transforms(self):
        """Rigid transformations between nodes."""
        return {tf_direction: transform.as_matrix(self.current_time) for
                tf_direction, transform in self._transforms.items()}

    def get_transform_at_time(self, from_frame, to_frame, time):
        """Request a transformation at a given time.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is requested in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        time : float
            Time at which we request the transformation.

        Returns
        -------
        A2B : array, shape (4, 4)
            Transformation from 'from_frame' to 'to_frame'

        Raises
        ------
        KeyError
            If one of the frames is unknown or there is no connection between
            them
        """
        previous_time = self._current_time
        self.set_time(time)

        A2B = self.get_transform(from_frame, to_frame)

        # revert internal state
        self.set_time(previous_time)
        return A2B

    def _transform_available(self, key):
        return key in self._transforms

    def _set_transform(self, key, A2B):
        self._transforms[key] = A2B

    def _get_transform(self, key):
        return self.transforms[key]

    def _del_transform(self, key):
        del self._transforms[key]

    def _check_transform(self, A2B):
        """Check validity of rigid transformation."""
        return A2B.check_transforms()
