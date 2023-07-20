import numpy as np

from .. import transformations as pt


def find_two_closest_indices(arr, target_value):
    # Calculate the absolute differences between the array elements and the target_value
    abs_diff = np.abs(arr - target_value)

    # Find the index of the minimum absolute difference (closest value to the target)
    closest_index = np.argmin(abs_diff)

    # Check if there is a match with the target_value
    if arr[closest_index] == target_value:
        return [closest_index, closest_index]

    # Set the selected index to a large value to exclude it from the search for the second closest
    abs_diff[closest_index] = np.inf

    # Find the index of the second minimum absolute difference (second closest value)
    second_closest_index = np.argmin(abs_diff)

    return [closest_index, second_closest_index]


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
    pq : array, shape (7,)
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