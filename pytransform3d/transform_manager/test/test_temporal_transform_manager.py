import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import (
    TemporalTransformManager,
    StaticTransform,
    NumpyTimeseriesTransform,
)


def create_sinusoidal_movement(
    duration_sec, sample_period, x_velocity, y_start_offset, start_time
):
    """Create a planar (z=0) sinusoidal movement around x-axis."""
    time_arr = np.arange(0, duration_sec, sample_period) + start_time
    N = len(time_arr)
    x_arr = np.linspace(0, x_velocity * duration_sec, N)

    spatial_freq = 1 / 5  # 1 sinus per 5m
    omega = 2 * np.pi * spatial_freq
    y_arr = np.sin(omega * x_arr)
    y_arr += y_start_offset

    dydx_arr = omega * np.cos(omega * x_arr)
    yaw_arr = np.arctan2(dydx_arr, np.ones_like(dydx_arr))

    pq_arr = list()
    for i in range(N):
        R = pr.active_matrix_from_extrinsic_euler_zyx([yaw_arr[i], 0, 0])
        T = pt.transform_from(R, [x_arr[i], y_arr[i], 0])
        pq = pt.pq_from_transform(T)
        pq_arr.append(pq)

    return time_arr, np.array(pq_arr)


def test_temporal_transform():
    rng = np.random.default_rng(0)
    A2B = pt.random_transform(rng)

    rng = np.random.default_rng(42)
    A2C = pt.random_transform(rng)

    tm = TemporalTransformManager()

    tm.add_transform("A", "B", StaticTransform(A2B))
    tm.add_transform("A", "C", StaticTransform(A2C))

    tm.current_time = 1234.0
    B2C = tm.get_transform("B", "C")

    C2B = tm.get_transform("C", "B")
    B2C_2 = pt.invert_transform(C2B)
    assert_array_almost_equal(B2C, B2C_2)

    B2C_3 = tm.get_transform_at_time("B", "C", 1234.0)
    assert_array_almost_equal(B2C_2, B2C_3)


def test_remove_frame_temporal_manager():
    """Test removing a frame from the transform manager."""
    tm = TemporalTransformManager()

    rng = np.random.default_rng(0)

    A2B = pt.random_transform(rng)
    A2D = pt.random_transform(rng)
    B2C = pt.random_transform(rng)
    D2E = pt.random_transform(rng)

    tm.add_transform("A", "B", StaticTransform(A2B))
    tm.add_transform("A", "D", StaticTransform(A2D))
    tm.add_transform("B", "C", StaticTransform(B2C))
    tm.add_transform("D", "E", StaticTransform(D2E))

    assert tm.has_frame("B")

    A2E = tm.get_transform("A", "E")

    # Check that connections are correctly represented in self.i and self.j
    assert tm.i == [
        tm.nodes.index("A"),
        tm.nodes.index("A"),
        tm.nodes.index("B"),
        tm.nodes.index("D"),
    ]
    assert tm.j == [
        tm.nodes.index("B"),
        tm.nodes.index("D"),
        tm.nodes.index("C"),
        tm.nodes.index("E"),
    ]

    tm.remove_frame("B")
    assert not tm.has_frame("B")

    # Ensure connections involving "B" are removed and the remaining
    # connections are correctly represented.
    assert tm.i == [tm.nodes.index("A"), tm.nodes.index("D")]
    assert tm.j == [tm.nodes.index("D"), tm.nodes.index("E")]

    with pytest.raises(KeyError, match="Unknown frame"):
        tm.get_transform("A", "B")
    with pytest.raises(KeyError, match="Unknown frame"):
        tm.get_transform("B", "C")

    assert tm.has_frame("A")
    assert tm.has_frame("C")
    assert tm.has_frame("D")
    assert tm.has_frame("E")

    # Ensure we cannot retrieve paths involving the removed frame
    with pytest.raises(KeyError, match="Cannot compute path"):
        tm.get_transform("A", "C")

    tm.get_transform("A", "D")
    tm.get_transform("D", "E")

    assert_array_almost_equal(A2E, tm.get_transform("A", "E"))


def test_internals():
    rng = np.random.default_rng(0)
    A2B = pt.random_transform(rng)

    rng = np.random.default_rng(42)
    A2C = pt.random_transform(rng)

    tm = TemporalTransformManager()

    tm.add_transform("A", "B", StaticTransform(A2B))
    tm.add_transform("A", "C", StaticTransform(A2C))

    tm.remove_transform("A", "C")
    assert ("A", "C") not in tm.transforms


def test_numpy_timeseries_transform():
    # create entities A and B together with their transformations from world
    duration = 10.0  # [s]
    sample_period = 0.5  # [s]
    velocity_x = 1  # [m/s]
    time_A, pq_arr_A = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=0.0, start_time=0.1
    )
    A2world = NumpyTimeseriesTransform(time_A, pq_arr_A)

    time_B, pq_arr_B = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=2.0, start_time=0.35
    )
    B2world = NumpyTimeseriesTransform(time_B, pq_arr_B)

    tm = TemporalTransformManager()

    tm.add_transform("A", "W", A2world)
    tm.add_transform("B", "W", B2world)

    query_time = time_A[0]  # Start time
    A2W_at_start = pt.transform_from_pq(pq_arr_A[0, :])
    A2W_at_start_2 = tm.get_transform_at_time("A", "W", query_time)
    assert A2W_at_start_2.ndim == 2
    assert_array_almost_equal(A2W_at_start, A2W_at_start_2, decimal=2)

    query_times = [time_A[0], time_A[0]]  # Start times
    A2Ws_at_start_2 = tm.get_transform_at_time("A", "W", query_times)
    assert A2Ws_at_start_2.ndim == 3
    assert_array_almost_equal(A2W_at_start, A2Ws_at_start_2[0], decimal=2)
    assert_array_almost_equal(A2W_at_start, A2Ws_at_start_2[1], decimal=2)

    A2Ws_at_start_2 = tm.get_transform_at_time("A", "W", query_times)
    assert_array_almost_equal(A2W_at_start, A2Ws_at_start_2[0], decimal=2)
    assert_array_almost_equal(A2W_at_start, A2Ws_at_start_2[1], decimal=2)
    assert A2Ws_at_start_2.ndim == 3

    query_time = 4.9  # [s]
    A2B_at_query_time = tm.get_transform_at_time("A", "B", query_time)

    origin_of_A_pos = pt.vector_to_point([0, 0, 0])
    origin_of_A_in_B_pos = pt.transform(A2B_at_query_time, origin_of_A_pos)
    origin_of_A_in_B_xyz = origin_of_A_in_B_pos[:-1]

    assert origin_of_A_in_B_xyz[0] == pytest.approx(-1.11, abs=1e-2)
    assert origin_of_A_in_B_xyz[1] == pytest.approx(-1.28, abs=1e-2)


def test_numpy_timeseries_transform_wrong_input_shapes():
    n_steps = 10
    with pytest.raises(ValueError, match="Number of timesteps"):
        time = np.arange(n_steps)
        pqs = np.random.randn(n_steps + 1, 7)
        NumpyTimeseriesTransform(time, pqs)

    with pytest.raises(ValueError, match="`pqs` matrix"):
        time = np.arange(10)
        pqs = np.random.randn(n_steps, 8)
        NumpyTimeseriesTransform(time, pqs)

    with pytest.raises(ValueError, match="Shape of PQ array"):
        time = np.arange(10)
        pqs = np.random.randn(n_steps, 8).flatten()
        NumpyTimeseriesTransform(time, pqs)


def test_numpy_timeseries_transform_multiple_query_times():
    # create entities A and B together with their transformations from world
    duration = 10.0  # [s]
    sample_period = 0.5  # [s]
    velocity_x = 1  # [m/s]
    time_A, pq_arr_A = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=0.0, start_time=0.1
    )
    A2world = NumpyTimeseriesTransform(time_A, pq_arr_A)

    time_B, pq_arr_B = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=2.0, start_time=0.35
    )
    B2world = NumpyTimeseriesTransform(time_B, pq_arr_B)

    tm = TemporalTransformManager()

    tm.add_transform("A", "W", A2world)
    tm.add_transform("B", "W", B2world)

    # test if shape is conserved correctly
    A2B_at_query_time = tm.get_transform_at_time("A", "B", 1.0)
    assert A2B_at_query_time.shape == (4, 4)

    A2B_at_query_time = tm.get_transform_at_time("A", "B", np.array([1.0]))
    assert A2B_at_query_time.shape == (1, 4, 4)

    query_times = np.array([4.9, 5.2])  # [s]
    A2B_at_query_time = tm.get_transform_at_time("A", "B", query_times)

    origin_of_A_pos = pt.vector_to_point([0, 0, 0])
    origin_of_A_in_B_pos1 = pt.transform(A2B_at_query_time[0], origin_of_A_pos)
    origin_of_A_in_B_xyz1 = origin_of_A_in_B_pos1[:-1]

    origin_of_A_in_B_x1, origin_of_A_in_B_y1 = (
        origin_of_A_in_B_xyz1[0],
        origin_of_A_in_B_xyz1[1],
    )

    assert origin_of_A_in_B_x1 == pytest.approx(-1.11, abs=1e-2)
    assert origin_of_A_in_B_y1 == pytest.approx(-1.28, abs=1e-2)


def test_temporal_transform_manager_incorrect_frame():
    duration = 10.0  # [s]
    sample_period = 0.5  # [s]
    velocity_x = 1  # [m/s]
    time_A, pq_arr_A = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=0.0, start_time=0.1
    )
    A2world = NumpyTimeseriesTransform(time_A, pq_arr_A)

    tm = TemporalTransformManager()
    tm.add_transform("A", "W", A2world)
    with pytest.raises(KeyError, match="Unknown frame"):
        tm.get_transform("B", "W")
    with pytest.raises(KeyError, match="Unknown frame"):
        tm.get_transform("A", "B")

    tm = TemporalTransformManager(check=False)
    tm.add_transform("A", "W", A2world)
    with pytest.raises(ValueError):
        tm.get_transform("B", "W")
    with pytest.raises(ValueError):
        tm.get_transform("A", "B")


def test_temporal_transform_manager_out_of_bounds():
    duration = 10.0  # [s]
    sample_period = 0.5  # [s]
    velocity_x = 1  # [m/s]
    time_A, pq_arr_A = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=0.0, start_time=0.0
    )
    A2world = NumpyTimeseriesTransform(time_A, pq_arr_A, time_clipping=True)

    time_B, pq_arr_B = create_sinusoidal_movement(
        duration, sample_period, velocity_x, y_start_offset=2.0, start_time=0.1
    )
    B2world = NumpyTimeseriesTransform(time_B, pq_arr_B, time_clipping=True)

    tm = TemporalTransformManager()
    tm.add_transform("A", "W", A2world)
    tm.add_transform("B", "W", B2world)

    assert min(time_A) == 0.0
    assert min(time_B) == 0.1
    A2B_at_start_time, A2B_before_start_time = tm.get_transform_at_time(
        "A", "B", [0.0, -0.1]
    )
    assert_array_almost_equal(A2B_at_start_time, A2B_before_start_time)

    assert max(time_A) == 9.5
    assert max(time_B) == 9.6
    A2B_at_end_time, A2B_after_end_time = tm.get_transform_at_time(
        "A", "B", [9.6, 10.0]
    )
    assert_array_almost_equal(A2B_at_end_time, A2B_after_end_time)

    A2world.time_clipping = False
    B2world.time_clipping = False

    with pytest.raises(
        ValueError, match="Query time at indices \[0\], time\(s\): \[0.\]"
    ):
        tm.get_transform_at_time("A", "B", [0.0, 0.1])

    with pytest.raises(
        ValueError,
        match=r"Query time at indices \[0 1\], time\(s\): \[ 9.7 10. \]",
    ):
        tm.get_transform_at_time("A", "B", [9.7, 10.0])
