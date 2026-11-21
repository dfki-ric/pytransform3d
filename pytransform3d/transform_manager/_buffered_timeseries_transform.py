"""Bounded transformation histories populated at run time."""

from collections import deque
from numbers import Integral

import numpy as np

from ._temporal_transform_manager import (
    NumpyTimeseriesTransform,
    TimeVaryingTransform,
)


class BufferedTimeseriesTransform(TimeVaryingTransform):
    """Append transformation samples to a bounded history.

    Samples must arrive in strictly increasing timestamp order. The oldest
    sample is discarded when the history reaches its capacity. Queries use
    the same screw linear interpolation (ScLERP) as
    :class:`NumpyTimeseriesTransform`.

    An empty history can be registered with a
    :class:`TemporalTransformManager`, but cannot be queried until a sample
    has been appended. A single sample supports queries at its timestamp, or
    at any timestamp if time clipping is enabled.

    Parameters
    ----------
    max_samples : int, optional (default: 1000)
        Positive maximum number of retained samples. Set this to 1 to retain
        only the most recent transformation.

    time_clipping : bool, optional (default: False)
        Return the oldest or newest retained pose for queries outside the
        retained interval. Otherwise, raise a ValueError.

    Notes
    -----
    Appending a sample takes constant time and memory is bounded by
    ``max_samples``. The first query after an append constructs an O(n)
    NumPy snapshot of the n retained samples. Further queries reuse that
    snapshot until the next append or clear. This class does not provide
    synchronization: callers must synchronize concurrent reads and writes.
    """

    def __init__(self, max_samples=1000, time_clipping=False):
        if (
            isinstance(max_samples, bool)
            or not isinstance(max_samples, Integral)
            or max_samples < 1
        ):
            raise ValueError("max_samples must be a positive integer.")
        self._time = deque(maxlen=int(max_samples))
        self._pqs = deque(maxlen=int(max_samples))
        self.time_clipping = time_clipping
        self._snapshot = None

    def __len__(self):
        """Return the number of retained samples."""
        return len(self._time)

    @property
    def time(self):
        """Copy of the retained timestamps, in increasing order."""
        return np.array(self._time, dtype=float)

    def append(self, time, pq):
        """Append a pose, discarding the oldest sample if necessary.

        Parameters
        ----------
        time : float
            Finite timestamp, strictly greater than the previous timestamp.

        pq : array-like, shape (7,)
            Position and quaternion (x, y, z, qw, qx, qy, qz). All entries
            must be finite and the quaternion must be nonzero. The sample
            is copied and its quaternion is normalized before storage.

        Returns
        -------
        self : BufferedTimeseriesTransform
            Updated history.

        Raises
        ------
        ValueError
            If the timestamp or pose is invalid. The history is unchanged
            when validation fails.
        """
        timestamp = np.asarray(time, dtype=float)
        if timestamp.ndim != 0 or not np.isfinite(timestamp):
            raise ValueError("time must be a finite scalar.")
        timestamp = float(timestamp)
        if self._time and timestamp <= self._time[-1]:
            raise ValueError("Timestamps must be strictly increasing.")
        sample = np.array(pq, dtype=float, copy=True)
        if sample.shape != (7,) or not np.all(np.isfinite(sample)):
            raise ValueError("pq must contain 7 finite values.")
        scale = np.max(np.abs(sample[3:]))
        if scale == 0.0:
            raise ValueError("The quaternion must be nonzero.")
        # Scale first to avoid overflow/underflow for non-unit inputs.
        sample[3:] /= scale
        sample[3:] /= np.linalg.norm(sample[3:])
        self._time.append(timestamp)
        self._pqs.append(sample)
        self._snapshot = None
        return self

    def as_matrix(self, query_time):
        """Interpolate a pose at one or more times.

        Parameters
        ----------
        query_time : float or array-like, shape (...,)
            Finite query time or times.

        Returns
        -------
        A2B : array, shape (..., 4, 4)
            Homogeneous matrices, or a single (4, 4) matrix for scalar input.

        Raises
        ------
        ValueError
            If the history is empty, a query is non-finite, or a query is
            outside the retained interval with time clipping disabled.
        """
        if not self._time:
            raise ValueError("Cannot query an empty transformation history.")
        times = np.asarray(query_time, dtype=float)
        if not np.all(np.isfinite(times)):
            raise ValueError("Query times must be finite.")
        if self._snapshot is None:
            self._snapshot = NumpyTimeseriesTransform(
                self.time, np.array(self._pqs), self.time_clipping
            )
        self._snapshot.time_clipping = self.time_clipping
        return self._snapshot.as_matrix(times.ravel()).reshape(
            times.shape + (4, 4)
        )

    def check_transforms(self):
        """Return this history; samples are validated when appended."""
        return self

    def clear(self):
        """Remove all samples and allow a new timestamp sequence.

        Returns
        -------
        self : BufferedTimeseriesTransform
            Empty history.
        """
        self._time.clear()
        self._pqs.clear()
        self._snapshot = None
        return self
