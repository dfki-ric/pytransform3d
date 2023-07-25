import abc

import numpy as np

from ._transform_graph_base import TransformGraphBase
from ..transformations import check_transform, transform_from_pq, \
    dual_quaternion_from_pq, pq_from_dual_quaternion, dual_quaternion_sclerp, \
    pq_from_transform


class TimeVaryingTransform(abc.ABC):
    """Time-varying rigid transformation."""

    @abc.abstractmethod
    def as_matrix(self, query_time):
        """Get transformation matrix at given time.

        Parameters
        ----------
        query_time : float
            Query time

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

    def as_matrix(self, query_time):
        return self._A2B

    def check_transforms(self):
        self._A2B = check_transform(self._A2B)
        return self


class NumpyTimeseriesTransform(TimeVaryingTransform):
    """Transformation sequence, represented in a numpy array.

    The interpolation is computed using screw linear interpolation (ScLERP)
    method.

    Parameters
    ----------
    time: np.ndarray, shape (N,)
        Numeric timesteps corresponding to the transformation samples.
        You can use unixtime, relative time (starting with 0.0).
    pqs : np.ndarray, shape (N, 7)
        Time-sequence of transformations, with each row representing a single
        sample as position-quarternion (PQ) structure.
    """

    def __init__(self, time, pqs):
        if len(pqs.shape) != 2:
            raise ValueError("Shape of PQ array must be 2-dimensional.")

        if time.size != pqs.shape[0]:
            raise ValueError("Number of timesteps is not equal to number of PQ"
                             "samples")

        if pqs.shape[1] != 7:
            raise ValueError("`pqs` matrix shall have 7 columns.")

        self._time_arr = time
        self._pqs = pqs

    def as_matrix(self, query_time):
        pq = pq = self._interpolate_pq_using_sclerp(query_time)
        return transform_from_pq(pq)

    def check_transforms(self):
        N = self._pqs.shape[0]
        for i in range(N):
            transform_at_i = transform_from_pq(self._pqs[i, :])
            checked_transfom = check_transform(transform_at_i)
            self._pqs[i, :] = pq_from_transform(checked_transfom)
        return self

    def _interpolate_pq_using_sclerp(self, query_time):

        # identify the index of the preceding sample
        t_arr = self._time_arr
        idx_timestep_earlier_wrt_query_time = \
            np.argmax(t_arr >= query_time) - 1

        # deal with first timestamp
        idx_timestep_earlier_wrt_query_time = max(
            idx_timestep_earlier_wrt_query_time, 0
        )

        pq_array = self._pqs

        # dual quaternion from preceding sample
        t_prev = t_arr[idx_timestep_earlier_wrt_query_time]
        pq_prev = pq_array[idx_timestep_earlier_wrt_query_time, :]
        dq_prev = dual_quaternion_from_pq(pq_prev)

        # dual quaternion from successive sample
        t_next = t_arr[idx_timestep_earlier_wrt_query_time + 1]
        pq_next = pq_array[idx_timestep_earlier_wrt_query_time + 1, :]
        dq_next = dual_quaternion_from_pq(pq_next)

        # since sclerp works with relative (0-1) positions
        rel_delta_t = (query_time - t_prev) / (t_next - t_prev)
        dq_interpolated = dual_quaternion_sclerp(dq_prev, dq_next, rel_delta_t)

        return pq_from_dual_quaternion(dq_interpolated)


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
