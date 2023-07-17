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
        super(TemporalTransformManager, self).__init__(strict_check, check)
    
    def get_transform(self, from_frame, to_frame, time):
        """Request a transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is requested in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        Returns
        -------
        A2B : Any
            Transformation from 'from_frame' to 'to_frame'

        Raises
        ------
        KeyError
            If one of the frames is unknown or there is no connection between
            them
        """
        if self.check:
            if from_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % from_frame)
            if to_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % to_frame)

        if (from_frame, to_frame) in self.transforms:
            return self._transforms[(from_frame, to_frame).as_matrix(time)]

        if (to_frame, from_frame) in self.transforms:
            return self._invert_transform(
                self._transforms[(to_frame, from_frame)].as_matrix(time))

        i = self.nodes.index(from_frame)
        j = self.nodes.index(to_frame)
        if not np.isfinite(self.dist[i, j]):
            raise KeyError("Cannot compute path from frame '%s' to "
                           "frame '%s'." % (from_frame, to_frame))

        path = self._shortest_path(i, j)
        return self._path_transform(path, time)
    
    @property
    def transforms(self):
        """Rigid transformations between nodes."""
        return self._transforms

    def _check_transform(self, A2B):
        return check_transform(A2B, strict_check=self.strict_check)

    def _invert_transform(self, A2B):
        return invert_transform(
            A2B, strict_check=self.strict_check, check=self.check)

    def _path_transform(self, path, time):
        A2B_matrix = np.eye(4)
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B_matrix = concat(A2B_matrix, self.get_transform(from_f, to_f, time),
                            strict_check=self.strict_check, check=self.check)
        return A2B
