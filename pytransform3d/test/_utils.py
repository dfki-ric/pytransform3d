import numpy as np

from .. import transformations as pt
from .. import rotations as pr


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
