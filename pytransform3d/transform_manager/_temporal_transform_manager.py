import abc

import numpy as np

from ._transform_graph_base import TransformGraphBase
from ._utils import find_two_closest_indices
from ..transformations import check_transform, transform_from_pq, \
    dual_quaternion_from_pq, pq_from_dual_quaternion, dual_quaternion_sclerp


class TimeVaryingTransform(abc.ABC):
    """Time-varying rigid transformation."""

    @abc.abstractmethod
    def as_matrix(self, time):
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
    def check_transforms(self):
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
    

class PandasTimeseriesTransform(TimeVaryingTransform):
    """Transformation, which does change over time, represented in a pandas
    dataframe.

    Parameters
    ----------
    df : dataframe, shape (N, 7)
        Time-sequence of transformations, with each row representing a single
        sample as position-quarternion (PQ) structure.
    """
    def __init__(self, df):
        self._df = df

    @property
    def _time_index(self):
        return self._df.index.to_numpy()
    
    def _check_if_time_is_in_bounds(self, time):
        time_index = self._time_index

        if (time < time_index[0]) or (time > time_index[-1]):
            raise ValueError("Time value is out of range!")

    def as_matrix(self, time):
        self._check_if_time_is_in_bounds(time)
        pq = self._interpolate_on_demand(time)
        return transform_from_pq(pq)
    
    def _interpolate_on_demand(self, time):
        indices = find_two_closest_indices(self._time_index, time)
        if indices[0] == indices[1]:
            # Match, no interpolation needed, we just take the sample!
            pq = self.pq_from_record(indices[0])
        else:
            pq = self._interpolate_pq(time)
        return pq

    def _interpolate_pq(self, time):
        t_arr = self._time_index

        pq_array = self._df[self.pq_column_names()].to_numpy()

        idx_timestep_earlier_wrt_query_time = np.argmax(t_arr >= time) - 1

        t_prev = t_arr[idx_timestep_earlier_wrt_query_time]
        pq_prev = pq_array[idx_timestep_earlier_wrt_query_time, :]
        dq_prev = dual_quaternion_from_pq(pq_prev)

        t_next = t_arr[idx_timestep_earlier_wrt_query_time + 1]
        pq_next = pq_array[idx_timestep_earlier_wrt_query_time + 1, :]
        dq_next = dual_quaternion_from_pq(pq_next)

        # since sclerp works with relative (0-1) positions
        rel_delta_t = (time - t_prev) / (t_next - t_prev)
        dq_interpolated = dual_quaternion_sclerp(dq_prev, dq_next, rel_delta_t)

        return pq_from_dual_quaternion(dq_interpolated)

    def check_transforms(self):
        # TODO: check if index is numeric
        return self
    
    @staticmethod
    def pq_column_names():
        return [f"p{axis}" for axis in "xyz"] + [f"q{axis}" for axis in "wxyz"]
    
    def pq_from_record(self, index: int) -> np.ndarray:
        transform_sample = self._df.iloc[index]
        pq = transform_sample[self.pq_column_names()].array
        return pq



class TemporalTransformManager(TransformGraphBase):
    """Manage time-varying transformations.

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
        self._transforms = {}
        self._current_time = 0.0

    @property
    def current_time(self):
        """Current time at which we evaluate transformations."""
        return self._current_time

    @current_time.setter
    def current_time(self, time):
        """Set current time at which we evaluate transformations."""
        self._current_time = time

    @property
    def transforms(self):
        """Rigid transformations between nodes."""
        return {transform_key: transform.as_matrix(self.current_time) for
                transform_key, transform in self._transforms.items()}

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
        previous_time = self.current_time
        self.current_time = time

        A2B = self.get_transform(from_frame, to_frame)

        # revert internal state
        self.current_time = previous_time
        return A2B

    def _transform_available(self, key):
        return key in self._transforms

    def _set_transform(self, key, A2B):
        self._transforms[key] = A2B

    def _get_transform(self, key):
        return self._transforms[key].as_matrix(self._current_time)

    def _del_transform(self, key):
        del self._transforms[key]

    def _check_transform(self, A2B):
        return A2B.check_transforms()
