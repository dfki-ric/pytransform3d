import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from pytransform3d.transform_manager import (
    NumpyTimeseriesTransform,
    StaticTransform,
    TemporalTransformManager,
    TimeVaryingTransform,
)


@pytest.fixture
def manager():
    pqs = np.array([[0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 1, 0, 0, 0]])
    manager = TemporalTransformManager()
    manager.add_transform(
        "sensor", "world", NumpyTimeseriesTransform([0.0, 2.0], pqs)
    )
    manager.add_transform("world", "map", StaticTransform(np.eye(4)))
    manager.add_transform("other", "disconnected", StaticTransform(np.eye(4)))
    return manager


@pytest.mark.parametrize("previous_time", [0.5, np.array([0.25, 0.75])])
@pytest.mark.parametrize(
    "from_frame,to_frame,query_time,error,message",
    [
        ("missing", "world", 1.0, KeyError, "Unknown frame"),
        ("sensor", "missing", 1.0, KeyError, "Unknown frame"),
        ("sensor", "other", 1.0, KeyError, "Cannot compute path"),
        ("sensor", "world", 3.0, ValueError, "out of range"),
        ("world", "sensor", 3.0, ValueError, "out of range"),
        ("sensor", "map", 3.0, ValueError, "out of range"),
    ],
)
def test_query_restores_time_on_error(
    manager, previous_time, from_frame, to_frame, query_time, error, message
):
    manager.current_time = previous_time
    before = manager.get_transform("sensor", "world")
    with pytest.raises(error, match=message):
        manager.get_transform_at_time(from_frame, to_frame, query_time)
    assert manager.current_time is previous_time
    assert_array_almost_equal(manager.get_transform("sensor", "world"), before)


@pytest.mark.parametrize("query_time", [1.0, np.array([0.5, 1.5])])
def test_query_restores_time_on_success(manager, query_time):
    previous_time = np.array([0.25, 0.75])
    manager.current_time = previous_time
    before = manager.get_transform("sensor", "world")
    actual = manager.get_transform_at_time("sensor", "world", query_time)
    expected = np.broadcast_to(np.eye(4), np.shape(query_time) + (4, 4)).copy()
    expected[..., 0, 3] = query_time
    assert_array_almost_equal(actual, expected)
    assert manager.current_time is previous_time
    assert_array_almost_equal(manager.get_transform("sensor", "world"), before)


def test_query_restores_time_when_transform_raises():
    failure = RuntimeError("transform evaluation failed")

    class FailingTransform(TimeVaryingTransform):
        def as_matrix(self, query_time):
            raise failure

        def check_transforms(self):
            return self

    manager = TemporalTransformManager()
    manager.add_transform("sensor", "world", FailingTransform())
    previous_time = np.array([0.25, 0.75])
    manager.current_time = previous_time
    with pytest.raises(RuntimeError) as caught:
        manager.get_transform_at_time("sensor", "world", 1.0)
    assert caught.value is failure
    assert manager.current_time is previous_time
