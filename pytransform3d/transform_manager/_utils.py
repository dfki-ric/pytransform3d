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
