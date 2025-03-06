"""Batch operations on rotations in three dimensions - SO(3).

Conversions from this module operate on batches of orientations or rotations
and can be orders of magnitude faster than a loop of individual conversions.

All functions operate on nd arrays, where the last dimension (vectors) or
the last two dimensions (matrices) contain individual rotations.
"""

from ._angle import (
    active_matrices_from_angles,
)
from ._axis_angle import (
    norm_axis_angles,
    matrices_from_compact_axis_angles,
)
from ._euler import (
    active_matrices_from_intrinsic_euler_angles,
    active_matrices_from_extrinsic_euler_angles,
)
from ._matrix import (
    axis_angles_from_matrices,
    quaternions_from_matrices,
)
from ._quaternion import (
    smooth_quaternion_trajectory,
    batch_concatenate_quaternions,
    batch_q_conj,
    quaternion_slerp_batch,
    axis_angles_from_quaternions,
    matrices_from_quaternions,
    batch_quaternion_wxyz_from_xyzw,
    batch_quaternion_xyzw_from_wxyz,
)
from ._utils import (
    norm_vectors,
    angles_between_vectors,
    cross_product_matrices,
)

__all__ = [
    "norm_vectors",
    "norm_axis_angles",
    "angles_between_vectors",
    "cross_product_matrices",
    "active_matrices_from_angles",
    "active_matrices_from_intrinsic_euler_angles",
    "active_matrices_from_extrinsic_euler_angles",
    "matrices_from_compact_axis_angles",
    "axis_angles_from_matrices",
    "quaternions_from_matrices",
    "smooth_quaternion_trajectory",
    "batch_concatenate_quaternions",
    "batch_q_conj",
    "quaternion_slerp_batch",
    "axis_angles_from_quaternions",
    "matrices_from_quaternions",
    "batch_quaternion_xyzw_from_wxyz",
    "batch_quaternion_wxyz_from_xyzw",
]
