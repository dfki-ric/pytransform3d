"""Trajectories in three dimensions - SE(3).

Conversions from this module operate on batches of poses or transformations
and can be 400 to 1000 times faster than a loop of individual conversions.
"""

from ._dual_quaternions import (
    batch_dq_conj,
    batch_dq_q_conj,
    batch_dq_prod_vector,
    batch_concatenate_dual_quaternions,
    dual_quaternions_power,
    dual_quaternions_sclerp,
    pqs_from_dual_quaternions,
    screw_parameters_from_dual_quaternions,
    transforms_from_dual_quaternions,
)
from ._plot import (
    plot_trajectory,
)
from ._pqs import (
    transforms_from_pqs,
    dual_quaternions_from_pqs,
)
from ._random import (
    random_trajectories,
)
from ._screws import (
    mirror_screw_axis_direction,
    transforms_from_exponential_coordinates,
    dual_quaternions_from_screw_parameters,
)
from ._transforms import (
    invert_transforms,
    concat_one_to_many,
    concat_many_to_one,
    concat_many_to_many,
    concat_dynamic,
    pqs_from_transforms,
    exponential_coordinates_from_transforms,
    dual_quaternions_from_transforms,
)

__all__ = [
    "invert_transforms",
    "concat_one_to_many",
    "concat_many_to_one",
    "concat_many_to_many",
    "concat_dynamic",
    "transforms_from_pqs",
    "pqs_from_transforms",
    "exponential_coordinates_from_transforms",
    "transforms_from_exponential_coordinates",
    "dual_quaternions_from_pqs",
    "dual_quaternions_from_transforms",
    "pqs_from_dual_quaternions",
    "screw_parameters_from_dual_quaternions",
    "dual_quaternions_from_screw_parameters",
    "dual_quaternions_power",
    "dual_quaternions_sclerp",
    "transforms_from_dual_quaternions",
    "batch_concatenate_dual_quaternions",
    "batch_dq_conj",
    "batch_dq_q_conj",
    "batch_dq_prod_vector",
    "mirror_screw_axis_direction",
    "random_trajectories",
    "plot_trajectory",
]
