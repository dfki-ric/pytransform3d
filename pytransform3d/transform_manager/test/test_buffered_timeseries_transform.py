import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import (
    BufferedTimeseriesTransform,
    NumpyTimeseriesTransform,
    StaticTransform,
    TemporalTransformManager,
)


def pose(x=0.0, angle=0.0):
    return np.r_[x, 0, 0, pr.quaternion_from_axis_angle([0, 0, 1, angle])]


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True, None, "3"])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        BufferedTimeseriesTransform(capacity)


def test_empty_history_can_be_registered_but_not_queried():
    history = BufferedTimeseriesTransform(np.int64(3))
    manager = TemporalTransformManager()
    assert history.check_transforms() is history
    manager.add_transform("sensor", "world", history)
    assert len(history) == 0
    assert history.time.shape == (0,)
    with pytest.raises(ValueError, match="empty"):
        manager.get_transform_at_time("sensor", "world", 0.0)


@pytest.mark.parametrize("clipping", [False, True])
def test_single_sample(clipping):
    history = BufferedTimeseriesTransform(time_clipping=clipping)
    history.append(5.0, pose(2.0, 0.7))
    expected = pt.transform_from_pq(pose(2.0, 0.7))
    assert_allclose(history.as_matrix(5.0), expected)
    assert history.as_matrix(5.0).shape == (4, 4)
    if clipping:
        assert_allclose(history.as_matrix([-1, 20]), [expected, expected])
    else:
        with pytest.raises(ValueError, match="out of range"):
            history.as_matrix(4.0)


@pytest.mark.parametrize("capacity", [1, 2, 5])
def test_streaming_queries_match_batch_sclerp_after_eviction(capacity):
    rng = np.random.default_rng(42)
    history = BufferedTimeseriesTransform(capacity)
    pqs = [pt.pq_from_transform(pt.random_transform(rng)) for _ in range(20)]
    for i, pq in enumerate(pqs):
        assert history.append(float(i), pq) is history
        start = max(0, i - capacity + 1)
        assert len(history) == i - start + 1
        assert_array_equal(history.time, np.arange(start, i + 1))
        reference = NumpyTimeseriesTransform(
            np.arange(start, i + 1), np.array(pqs[start : i + 1])
        )
        queries = np.linspace(start, i, 9)
        assert_allclose(
            history.as_matrix(queries), reference.as_matrix(queries)
        )
        assert_allclose(history.as_matrix(float(i)), pt.transform_from_pq(pq))
    if capacity < 20:
        with pytest.raises(ValueError, match="out of range"):
            history.as_matrix(0.0)


def test_query_shapes_clipping_and_latest_only():
    history = BufferedTimeseriesTransform(1, time_clipping=True)
    history.append(0, pose(0)).append(1, pose(1))
    queries = np.array([[-2, 1], [4, 10]])
    expected = np.broadcast_to(pt.transform_from_pq(pose(1)), (2, 2, 4, 4))
    assert_allclose(history.as_matrix(queries), expected)
    assert history.as_matrix(np.empty((2, 0))).shape == (2, 0, 4, 4)
    history.time_clipping = False
    with pytest.raises(ValueError, match="out of range"):
        history.as_matrix(queries)


def test_multidimensional_interpolation():
    history = BufferedTimeseriesTransform()
    history.append(0, pose(0)).append(2, pose(2))
    queries = np.array([[0, 0.5], [1, 2]])
    result = history.as_matrix(queries)
    assert result.shape == (2, 2, 4, 4)
    assert_allclose(result[..., 0, 3], queries)


@pytest.mark.parametrize("time", [np.nan, np.inf, -np.inf, [1.0]])
def test_invalid_timestamps_leave_history_unchanged(time):
    history = BufferedTimeseriesTransform(1).append(0, pose())
    with pytest.raises(ValueError, match="finite scalar"):
        history.append(time, pose(1))
    assert_array_equal(history.time, [0])


@pytest.mark.parametrize("time", [0.0, -1.0])
def test_nonincreasing_timestamps_leave_history_unchanged(time):
    history = BufferedTimeseriesTransform(1).append(0, pose())
    with pytest.raises(ValueError, match="strictly increasing"):
        history.append(time, pose(1))
    assert_array_equal(history.time, [0])


@pytest.mark.parametrize(
    "pq",
    [np.zeros(6), np.zeros((1, 7)), np.full(7, np.nan), np.full(7, np.inf)],
)
def test_invalid_pose_leaves_history_unchanged(pq):
    history = BufferedTimeseriesTransform(1).append(0, pose())
    with pytest.raises(ValueError, match="7 finite values"):
        history.append(1, pq)
    assert_array_equal(history.time, [0])
    assert_allclose(history.as_matrix(0), np.eye(4))


def test_zero_quaternion_rejected_without_eviction():
    history = BufferedTimeseriesTransform(1).append(0, pose())
    with pytest.raises(ValueError, match="nonzero"):
        history.append(1, np.zeros(7))
    assert_array_equal(history.time, [0])


@pytest.mark.parametrize("scale", [2.0, 1e300, 1e-300])
def test_samples_are_copied_and_quaternions_normalized(scale):
    sample = pose(2, 0.9)
    expected = pt.transform_from_pq(sample)
    sample[3:] *= scale
    history = BufferedTimeseriesTransform().append(0, sample)
    sample[:] = 0
    times = history.time
    times[:] = 123
    assert_array_equal(history.time, [0])
    assert_allclose(history.as_matrix(0), expected, atol=1e-15)


@pytest.mark.parametrize("query", [np.nan, np.inf, [0, np.nan]])
def test_nonfinite_queries(query):
    history = BufferedTimeseriesTransform().append(0, pose())
    with pytest.raises(ValueError, match="finite"):
        history.as_matrix(query)


def test_append_and_clear_invalidate_cached_queries():
    history = BufferedTimeseriesTransform(time_clipping=True).append(0, pose())
    assert_allclose(history.as_matrix(10), np.eye(4))
    history.append(1, pose(3))
    assert_allclose(history.as_matrix(10), pt.transform_from_pq(pose(3)))
    assert history.clear() is history
    assert len(history) == 0
    with pytest.raises(ValueError, match="empty"):
        history.as_matrix(0)
    history.append(-2, pose(-3))
    assert_allclose(history.as_matrix(-2), pt.transform_from_pq(pose(-3)))


def test_temporal_manager_sees_appends_without_reregistering():
    history = BufferedTimeseriesTransform(2)
    manager = TemporalTransformManager()
    manager.add_transform("sensor", "robot", history)
    offset = pt.transform_from_pq(pose(10))
    manager.add_transform("robot", "world", StaticTransform(offset))
    history.append(0, pose()).append(2, pose(2))
    expected = pt.transform_from_pq(pose(11))
    assert_allclose(
        manager.get_transform_at_time("sensor", "world", 1), expected
    )
    assert_allclose(
        manager.get_transform_at_time("world", "sensor", 1),
        pt.invert_transform(expected),
    )
    history.append(4, pose(4))
    assert_allclose(
        manager.get_transform_at_time("sensor", "world", 3),
        pt.transform_from_pq(pose(13)),
    )
    with pytest.raises(ValueError, match="out of range"):
        manager.get_transform_at_time("sensor", "world", 0)
