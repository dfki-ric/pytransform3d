import abc

import numpy as np

from ..batch_rotations import norm_vectors

from ._transform_graph_base import TransformGraphBase
from ..transformations import check_transform
from ..trajectories import (
    dual_quaternions_from_pqs,
    pqs_from_dual_quaternions,
    dual_quaternions_sclerp,
    transforms_from_pqs,
    concat_dynamic,
    invert_transforms
)


class TimeVaryingTransform(abc.ABC):
    """Time-varying rigid transformation.

    You have to inherit from this abstract base class to use the
    TemporalTransformManager. Two implementations of the interface that
    are already available are :class:`StaticTransform` and
    :class:`NumpyTimeseriesTransform`.
    """

    @abc.abstractmethod
    def as_matrix(self, query_time):
        """Get transformation matrix at given time.

        Parameters
        ----------
        query_time : Union[float, array-like shape (n_queries,)]
            Query time

        Returns
        -------
        A2B_t : array, shape (4, 4) or (n_queries, 4, 4)
            Homogeneous transformation matrix at given time or times
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
    time: array, shape (n_steps,)
        Numeric timesteps corresponding to the transformation samples.
        You can use, for example, unix timestamps, relative time (starting
        with 0).

    pqs : array, shape (n_steps, 7)
        Time-sequence of transformations, with each row representing a single
        sample as position-quarternion (PQ) structure.

    time_clipping : bool, optional (default: False)
        Clip time to minimum or maximum respectively when query time is out of
        range of the time series. If this deactivated, we will raise a
        ValueError when the query time is out of range.
    """
    def __init__(self, time, pqs, time_clipping=False):
        self.time = np.asarray(time)
        self._pqs = np.asarray(pqs)
        self.time_clipping = time_clipping

        if len(self._pqs.shape) != 2:
            raise ValueError("Shape of PQ array must be 2-dimensional.")

        if self.time.size != self._pqs.shape[0]:
            raise ValueError(
                "Number of timesteps does not equal to number of PQ samples")

        if self._pqs.shape[1] != 7:
            raise ValueError("`pqs` matrix shall have 7 columns.")

        self._min_time = min(self.time)
        self._max_time = max(self.time)

    def as_matrix(self, query_time):
        """Get transformation matrix at given time.

        Parameters
        ----------
        query_time : Union[float, array-like shape (...)]
            Query time

        Returns
        -------
        A2B_t : array, shape (4, 4) or (..., 4, 4)
            Homogeneous transformation matrix / matrices at given time / times.
        """
        query_time = np.asarray(query_time)
        pq = self._interpolate_pq_using_sclerp(np.atleast_1d(query_time))
        transforms = transforms_from_pqs(pq)
        output_shape = list(query_time.shape) + [4, 4]
        return transforms.reshape(*output_shape)

    def check_transforms(self):
        self._pqs[:, 3:] = norm_vectors(self._pqs[:, 3:])
        return self

    def _interpolate_pq_using_sclerp(self, query_time_arr):
        # identify the index of the preceding sample
        idxs_timestep_earlier_wrt_query_time = np.searchsorted(
            self.time, query_time_arr, side='right'
        ) - 1

        # deal with first and last timestamp
        before_start = query_time_arr < self._min_time
        after_end = query_time_arr > self._max_time
        out_of_range = np.logical_or(before_start, after_end)
        if self.time_clipping:
            min_index = 0
            max_index = self.time.shape[0] - 2
            idxs_timestep_earlier_wrt_query_time = np.clip(
                idxs_timestep_earlier_wrt_query_time, min_index, max_index)
        elif any(out_of_range):
            indices = np.where(out_of_range)[0]
            times = query_time_arr[indices]
            raise ValueError(
                "Query time at indices %s, time(s): %s is/are out of range of "
                "time series." % (indices, times))

        idxs_timestep_later_wrt_query_time = \
            idxs_timestep_earlier_wrt_query_time + 1
        if self.time_clipping:
            before_or_eq_start = query_time_arr <= self._min_time
            after_or_eq_end = query_time_arr >= self._max_time
            idxs_timestep_later_wrt_query_time[before_or_eq_start] = \
                idxs_timestep_earlier_wrt_query_time[before_or_eq_start]
            idxs_timestep_earlier_wrt_query_time[after_or_eq_end] = \
                idxs_timestep_later_wrt_query_time[after_or_eq_end]

        # dual quaternion from preceding sample
        times_prev = self.time[idxs_timestep_earlier_wrt_query_time]
        pqs_prev = self._pqs[idxs_timestep_earlier_wrt_query_time]
        dqs_prev = dual_quaternions_from_pqs(pqs_prev)

        # dual quaternion from successive sample
        times_next = self.time[idxs_timestep_later_wrt_query_time]
        pqs_next = self._pqs[idxs_timestep_later_wrt_query_time]
        dqs_next = dual_quaternions_from_pqs(pqs_next)

        # For each query timestamp, determine its normalized time distance
        # between the preceeding and succeeding sample, since ScLERP works with
        # relative times t in [0, 1]. If query time equals a time sample
        # exactly, 0.0 is set.
        rel_times_between_samples = np.zeros_like(query_time_arr, dtype=float)
        interpolation_case = times_prev != times_next
        rel_times_between_samples[interpolation_case] = (
            query_time_arr[interpolation_case] - times_prev[interpolation_case]
        ) / (times_next[interpolation_case] - times_prev[interpolation_case])
        dqs_interpolated = dual_quaternions_sclerp(
            dqs_prev, dqs_next, rel_times_between_samples)
        return pqs_from_dual_quaternions(dqs_interpolated)


class TemporalTransformManager(TransformGraphBase):
    """Manage time-varying transformations.

    See :ref:`transformations_over_time` for more information.

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
        self._current_time = np.array([0.0])

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

        time : Union[float, array-like shape (...)]
            Time or times at which we request the transformation. If the query
            time is out of bounds, it will be clipped to either the first or
            last available time.

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

    def get_transform(self, from_frame, to_frame):
        """Request a transformation.
        
        The internal current_time will be used for time based transformations.
        If the query time is out of bounds, it will be clipped to either the
        first or the last available time.

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

        if self._transform_available((from_frame, to_frame)):
            return self._get_transform((from_frame, to_frame))

        if self._transform_available((to_frame, from_frame)):
            return invert_transforms(
                self._get_transform((to_frame, from_frame))
            )

        i = self.nodes.index(from_frame)
        j = self.nodes.index(to_frame)
        if not np.isfinite(self.dist[i, j]):
            raise KeyError(
                "Cannot compute path from frame '%s' to frame '%s'."
                % (from_frame, to_frame)
            )

        path = self._shortest_path(i, j)
        return self._path_transform(path)

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

    def _path_transform(self, path):
        """Convert sequence of node names to rigid transformation."""
        A2B = np.eye(4)
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B = concat_dynamic(A2B, self.get_transform(from_f, to_f))
        return A2B
