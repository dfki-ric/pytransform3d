"""
==================================
Managing Transformations over Time
==================================

In this example, given two tranformation trajectories, we will interpolate both
and use the transform manager for the target timestep.
"""
import numpy as np

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager


def create_sinusoidal_movement(
    duration_sec, sample_period, x_velocity, y_start_offset, start_time
):
    """Create a planar (z=0) sinusoidal movement around x-axis."""

    time_arr = np.arange(0, duration_sec, sample_period) + start_time
    N = len(time_arr)
    x_arr = np.linspace(0, x_velocity * duration_sec, N)

    spatial_freq = 1 / 5  # 1 sinus per 5m
    y_arr = np.sin(2 * np.pi * spatial_freq * x_arr)
    y_arr += y_start_offset

    dydx_arr = np.cos(2 * np.pi * spatial_freq * x_arr)
    yaw_arr = np.arctan2(dydx_arr, np.ones_like(dydx_arr))

    pq_arr = list()
    for i in range(N):
        R = pr.active_matrix_from_extrinsic_euler_zyx([yaw_arr[i], 0, 0])
        T = pt.transform_from(R, [x_arr[i], y_arr[i], 0])
        pq = pt.pq_from_transform(T)
        pq_arr.append(pq)

    return time_arr, np.array(pq_arr)


# create entities A and B
duration = 10.0  # [s]
sample_period = 0.5  # [s]
velocity_x = 1  # [m/s]
time_W2A, pq_arr_W2A = create_sinusoidal_movement(
    duration, sample_period, velocity_x, y_start_offset=0.0, start_time=0.1
)
time_W2B, pq_arr_W2B = create_sinusoidal_movement(
    duration, sample_period, velocity_x, y_start_offset=2.0, start_time=0.35
)


def interpolate_pq(query_time, t_arr, pq_array):
    """Interpolate a transformation trajectory at a target time.

    Parameters
    ----------
    query_time : float
        Target timestamp
    t_arr : array-like, shape (N,)
        Timesteps from the transformation trajectory
    pq_array : array-like, shape (N,7)
        Transformation trajectory with each row representing
        position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    Returns
    -------
    pq : array-like, shape (7,)
        Interpolated position and orientation quaternion:
        (x, y, z, qw, qx, qy, qz)
    """

    idx_timestep_earlier_wrt_query_time = np.argmax(t_arr >= query_time) - 1

    t_prev = t_arr[idx_timestep_earlier_wrt_query_time]
    pq_prev = pq_array[idx_timestep_earlier_wrt_query_time, :]
    dq_prev = pt.dual_quaternion_from_pq(pq_prev)

    t_next = t_arr[idx_timestep_earlier_wrt_query_time + 1]
    pq_next = pq_array[idx_timestep_earlier_wrt_query_time + 1, :]
    dq_next = pt.dual_quaternion_from_pq(pq_next)

    # since sclerp works with relative (0-1) positions
    rel_delta_t = (query_time - t_prev) / (t_next - t_prev)
    dq_interpolated = pt.dual_quaternion_sclerp(dq_prev, dq_next, rel_delta_t)

    return pt.pq_from_dual_quaternion(dq_interpolated)


query_time = 5.0
pq_W2A = interpolate_pq(query_time, time_W2A, pq_arr_W2A)
pq_W2B = interpolate_pq(query_time, time_W2B, pq_arr_W2B)

# now we can feed the transform manager
tm = TransformManager()
tm.add_transform("world", "A", pt.transform_from_pq(pq_W2A))
tm.add_transform("world", "B", pt.transform_from_pq(pq_W2B))

query_transform_A2B = tm.get_transform("A", "B")

origin_of_B_in_A = pt.transform(query_transform_A2B,
                                pt.vector_to_point([0, 0, 0]))
origin_of_B_in_A = origin_of_B_in_A[:-1]

assert origin_of_B_in_A[0] < 0  # x
assert origin_of_B_in_A[1] > 0  # y
assert origin_of_B_in_A[2] == 0  # z
