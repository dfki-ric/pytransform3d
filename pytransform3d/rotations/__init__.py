"""Rotations in three dimensions - SO(3).

See :doc:`user_guide/rotations` for more information.
"""
from ._constants import (
    eps, unitx, unity, unitz, q_i, q_j, q_k, q_id, a_id, R_id, p0)
from ._utils import (
    norm_angle, norm_vector, angle_between_vectors, perpendicular_to_vector,
    vector_projection, perpendicular_to_vectors,
    norm_axis_angle, compact_axis_angle_near_pi, norm_compact_axis_angle,
    matrix_requires_renormalization, norm_matrix,
    quaternion_requires_renormalization,
    plane_basis_from_normal,
    check_skew_symmetric_matrix, check_matrix, check_quaternion,
    check_quaternions, check_axis_angle, check_compact_axis_angle,
    check_rotor, check_mrp)
from ._random import (
    random_vector,
    random_axis_angle,
    random_compact_axis_angle,
    random_quaternion)
from ._conversions import (
    quaternion_from_axis_angle, quaternion_from_compact_axis_angle,
    quaternion_from_matrix, quaternion_wxyz_from_xyzw,
    quaternion_xyzw_from_wxyz, quaternion_from_extrinsic_euler_xyz,
    axis_angle_from_quaternion, axis_angle_from_compact_axis_angle,
    axis_angle_from_matrix, axis_angle_from_two_directions,
    compact_axis_angle, compact_axis_angle_from_quaternion,
    compact_axis_angle_from_matrix,
    matrix_from_quaternion, matrix_from_compact_axis_angle,
    matrix_from_axis_angle, matrix_from_two_vectors,
    active_matrix_from_angle, norm_euler, euler_near_gimbal_lock,
    matrix_from_euler,
    active_matrix_from_extrinsic_euler_xyx,
    active_matrix_from_intrinsic_euler_xyx,
    active_matrix_from_extrinsic_euler_xyz,
    active_matrix_from_extrinsic_euler_xzx,
    active_matrix_from_extrinsic_euler_xzy,
    active_matrix_from_extrinsic_euler_yxy,
    active_matrix_from_extrinsic_euler_yxz,
    active_matrix_from_extrinsic_euler_yzx,
    active_matrix_from_extrinsic_euler_yzy,
    active_matrix_from_extrinsic_euler_zxy,
    active_matrix_from_extrinsic_euler_zxz,
    active_matrix_from_extrinsic_euler_zyz,
    active_matrix_from_intrinsic_euler_xyz,
    active_matrix_from_intrinsic_euler_xzx,
    active_matrix_from_extrinsic_euler_zyx,
    active_matrix_from_intrinsic_euler_xzy,
    active_matrix_from_intrinsic_euler_yxy,
    active_matrix_from_intrinsic_euler_yxz,
    active_matrix_from_intrinsic_euler_yzx,
    active_matrix_from_intrinsic_euler_yzy,
    active_matrix_from_intrinsic_euler_zxy,
    active_matrix_from_intrinsic_euler_zxz,
    active_matrix_from_intrinsic_euler_zyx,
    active_matrix_from_intrinsic_euler_zyz,
    active_matrix_from_extrinsic_roll_pitch_yaw,
    passive_matrix_from_angle,
    intrinsic_euler_xyz_from_active_matrix,
    intrinsic_euler_xyx_from_active_matrix,
    intrinsic_euler_xzx_from_active_matrix,
    intrinsic_euler_xzy_from_active_matrix,
    intrinsic_euler_yxy_from_active_matrix,
    intrinsic_euler_yxz_from_active_matrix,
    intrinsic_euler_yzx_from_active_matrix,
    intrinsic_euler_yzy_from_active_matrix,
    intrinsic_euler_zxy_from_active_matrix,
    intrinsic_euler_zxz_from_active_matrix,
    intrinsic_euler_zyx_from_active_matrix,
    intrinsic_euler_zyz_from_active_matrix,
    extrinsic_euler_xyx_from_active_matrix,
    extrinsic_euler_xyz_from_active_matrix,
    extrinsic_euler_xzx_from_active_matrix,
    extrinsic_euler_xzy_from_active_matrix,
    extrinsic_euler_yxy_from_active_matrix,
    extrinsic_euler_yxz_from_active_matrix,
    extrinsic_euler_yzx_from_active_matrix,
    extrinsic_euler_yzy_from_active_matrix,
    extrinsic_euler_zxy_from_active_matrix,
    extrinsic_euler_zxz_from_active_matrix,
    extrinsic_euler_zyx_from_active_matrix,
    extrinsic_euler_zyz_from_active_matrix,
    euler_from_matrix,
    euler_from_quaternion,
    quaternion_from_angle,
    cross_product_matrix,
    mrp_from_quaternion,
    quaternion_from_mrp,
    mrp_from_axis_angle,
    axis_angle_from_mrp)
from ._quaternions import (
    quaternion_double, quaternion_integrate, quaternion_gradient,
    concatenate_quaternions, q_conj, q_prod_vector, quaternion_diff,
    quaternion_dist, quaternion_from_euler)
from ._mrp import (
    mrp_near_singularity, norm_mrp, mrp_double, concatenate_mrp,
    mrp_prod_vector)
from ._slerp import (slerp_weights, pick_closest_quaternion, quaternion_slerp,
                     axis_angle_slerp, rotor_slerp)
from ._testing import (
    assert_euler_equal, assert_quaternion_equal, assert_axis_angle_equal,
    assert_compact_axis_angle_equal, assert_rotation_matrix, assert_mrp_equal)
from ._plot import plot_basis, plot_axis_angle, plot_bivector
from ._rotors import (
    wedge, geometric_product, rotor_apply, rotor_reverse, concatenate_rotors,
    rotor_from_plane_angle, rotor_from_two_directions, matrix_from_rotor,
    plane_normal_from_bivector)
from ._jacobians import (
    left_jacobian_SO3, left_jacobian_SO3_series, left_jacobian_SO3_inv,
    left_jacobian_SO3_inv_series)

__all__ = [
    "eps",
    "unitx",
    "unity",
    "unitz",
    "q_i",
    "q_j",
    "q_k",
    "q_id",
    "a_id",
    "R_id",
    "p0",
    "norm_angle",
    "norm_vector",
    "angle_between_vectors",
    "perpendicular_to_vector",
    "vector_projection",
    "perpendicular_to_vectors",
    "norm_axis_angle",
    "compact_axis_angle_near_pi",
    "norm_compact_axis_angle",
    "matrix_requires_renormalization",
    "norm_matrix",
    "quaternion_requires_renormalization",
    "random_vector",
    "random_axis_angle",
    "random_compact_axis_angle",
    "random_quaternion",
    "check_skew_symmetric_matrix",
    "check_matrix",
    "check_quaternion",
    "check_quaternions",
    "check_axis_angle",
    "check_rotor",
    "check_compact_axis_angle",
    "check_mrp",
    "quaternion_from_axis_angle",
    "quaternion_from_compact_axis_angle",
    "quaternion_from_matrix",
    "quaternion_wxyz_from_xyzw",
    "quaternion_xyzw_from_wxyz",
    "quaternion_from_extrinsic_euler_xyz",
    "axis_angle_from_quaternion",
    "axis_angle_from_compact_axis_angle",
    "axis_angle_from_matrix",
    "axis_angle_from_two_directions",
    "compact_axis_angle",
    "compact_axis_angle_from_quaternion",
    "compact_axis_angle_from_matrix",
    "matrix_from_quaternion",
    "matrix_from_compact_axis_angle",
    "matrix_from_axis_angle",
    "matrix_from_two_vectors",
    "active_matrix_from_angle",
    "norm_euler",
    "euler_near_gimbal_lock",
    "matrix_from_euler",
    "active_matrix_from_extrinsic_euler_xyx",
    "active_matrix_from_intrinsic_euler_xyx",
    "active_matrix_from_extrinsic_euler_xyz",
    "active_matrix_from_extrinsic_euler_xzx",
    "active_matrix_from_extrinsic_euler_xzy",
    "active_matrix_from_extrinsic_euler_yxy",
    "active_matrix_from_extrinsic_euler_yxz",
    "active_matrix_from_extrinsic_euler_yzx",
    "active_matrix_from_extrinsic_euler_yzy",
    "active_matrix_from_extrinsic_euler_zxy",
    "active_matrix_from_extrinsic_euler_zxz",
    "active_matrix_from_extrinsic_euler_zyz",
    "active_matrix_from_intrinsic_euler_xyz",
    "active_matrix_from_intrinsic_euler_xzx",
    "active_matrix_from_extrinsic_euler_zyx",
    "active_matrix_from_intrinsic_euler_xzy",
    "active_matrix_from_intrinsic_euler_yxy",
    "active_matrix_from_intrinsic_euler_yxz",
    "active_matrix_from_intrinsic_euler_yzx",
    "active_matrix_from_intrinsic_euler_yzy",
    "active_matrix_from_intrinsic_euler_zxy",
    "active_matrix_from_intrinsic_euler_zxz",
    "active_matrix_from_intrinsic_euler_zyx",
    "active_matrix_from_intrinsic_euler_zyz",
    "active_matrix_from_extrinsic_roll_pitch_yaw",
    "passive_matrix_from_angle",
    "intrinsic_euler_xyz_from_active_matrix",
    "intrinsic_euler_xyx_from_active_matrix",
    "intrinsic_euler_xzx_from_active_matrix",
    "intrinsic_euler_xzy_from_active_matrix",
    "intrinsic_euler_yxy_from_active_matrix",
    "intrinsic_euler_yxz_from_active_matrix",
    "intrinsic_euler_yzx_from_active_matrix",
    "intrinsic_euler_yzy_from_active_matrix",
    "intrinsic_euler_zxy_from_active_matrix",
    "intrinsic_euler_zxz_from_active_matrix",
    "intrinsic_euler_zyx_from_active_matrix",
    "intrinsic_euler_zyz_from_active_matrix",
    "extrinsic_euler_xyx_from_active_matrix",
    "extrinsic_euler_xyz_from_active_matrix",
    "extrinsic_euler_xzx_from_active_matrix",
    "extrinsic_euler_xzy_from_active_matrix",
    "extrinsic_euler_yxy_from_active_matrix",
    "extrinsic_euler_yxz_from_active_matrix",
    "extrinsic_euler_yzx_from_active_matrix",
    "extrinsic_euler_yzy_from_active_matrix",
    "extrinsic_euler_zxy_from_active_matrix",
    "extrinsic_euler_zxz_from_active_matrix",
    "extrinsic_euler_zyx_from_active_matrix",
    "extrinsic_euler_zyz_from_active_matrix",
    "euler_from_matrix",
    "euler_from_quaternion",
    "quaternion_from_angle",
    "quaternion_from_euler",
    "mrp_near_singularity",
    "norm_mrp",
    "mrp_double",
    "concatenate_mrp",
    "mrp_prod_vector",
    "cross_product_matrix",
    "mrp_from_quaternion",
    "quaternion_from_mrp",
    "mrp_from_axis_angle",
    "axis_angle_from_mrp",
    "quaternion_double",
    "quaternion_integrate",
    "quaternion_gradient",
    "concatenate_quaternions",
    "q_conj",
    "q_prod_vector",
    "quaternion_diff",
    "quaternion_dist",
    "slerp_weights",
    "pick_closest_quaternion",
    "quaternion_slerp",
    "axis_angle_slerp",
    "assert_euler_equal",
    "assert_quaternion_equal",
    "assert_axis_angle_equal",
    "assert_compact_axis_angle_equal",
    "assert_rotation_matrix",
    "assert_mrp_equal",
    "plot_basis",
    "plot_axis_angle",
    "wedge",
    "geometric_product",
    "wedge",
    "geometric_product",
    "rotor_apply",
    "rotor_reverse",
    "concatenate_rotors",
    "rotor_from_plane_angle",
    "rotor_from_two_directions",
    "matrix_from_rotor",
    "plot_bivector",
    "rotor_slerp",
    "plane_normal_from_bivector",
    "plane_basis_from_normal",
    "left_jacobian_SO3",
    "left_jacobian_SO3_series",
    "left_jacobian_SO3_inv",
    "left_jacobian_SO3_inv_series"
]
