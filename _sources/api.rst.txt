.. _api:

=================
API Documentation
=================

This is the detailed documentation of all public classes and functions.
You can also search for specific modules, classes, or functions in the
:ref:`genindex`.


:mod:`pytransform3d.rotations`
==============================

.. automodule:: pytransform3d.rotations
    :no-members:
    :no-inherited-members:

Rotation Matrix
---------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_matrix

   ~matrix_requires_renormalization
   ~norm_matrix

   ~random_matrix

   ~plot_basis

   ~assert_rotation_matrix

   ~passive_matrix_from_angle
   ~active_matrix_from_angle
   ~matrix_from_two_vectors

   ~matrix_from_euler
   ~matrix_from_axis_angle
   ~matrix_from_compact_axis_angle
   ~matrix_from_quaternion
   ~matrix_from_rotor

Euler Angles
------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~euler_near_gimbal_lock
   ~norm_euler

   ~assert_euler_equal

   ~euler_from_quaternion
   ~euler_from_matrix

Axis-Angle
----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_axis_angle
   ~check_compact_axis_angle

   ~compact_axis_angle_near_pi
   ~norm_axis_angle
   ~norm_compact_axis_angle

   ~random_axis_angle
   ~random_compact_axis_angle

   ~plot_axis_angle

   ~assert_axis_angle_equal
   ~assert_compact_axis_angle_equal

   ~axis_angle_slerp

   ~axis_angle_from_two_directions

   ~axis_angle_from_matrix
   ~axis_angle_from_quaternion
   ~axis_angle_from_compact_axis_angle
   ~axis_angle_from_mrp
   ~compact_axis_angle
   ~compact_axis_angle_from_matrix
   ~compact_axis_angle_from_quaternion

Logarithm of Rotation
---------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_rot_log
   ~check_skew_symmetric_matrix

   ~rot_log_from_compact_axis_angle
   ~cross_product_matrix


Quaternion
----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_quaternion
   ~check_quaternions

   ~quaternion_requires_renormalization

   ~quaternion_double
   ~pick_closest_quaternion

   ~random_quaternion

   ~assert_quaternion_equal

   ~concatenate_quaternions
   ~q_prod_vector
   ~q_conj
   ~quaternion_slerp
   ~quaternion_dist
   ~quaternion_diff
   ~quaternion_gradient
   ~quaternion_integrate
   ~quaternion_from_angle
   ~quaternion_from_euler
   ~quaternion_from_matrix
   ~quaternion_from_axis_angle
   ~quaternion_from_compact_axis_angle
   ~quaternion_from_mrp
   ~quaternion_xyzw_from_wxyz
   ~quaternion_wxyz_from_xyzw

Rotor
-----

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_rotor

   ~plot_bivector

   ~wedge
   ~plane_normal_from_bivector
   ~geometric_product
   ~concatenate_rotors
   ~rotor_reverse
   ~rotor_apply
   ~rotor_slerp

   ~rotor_from_two_directions
   ~rotor_from_plane_angle

Modified Rodrigues Parameters
-----------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_mrp

   ~mrp_near_singularity
   ~norm_mrp

   ~assert_mrp_equal

   ~mrp_double

   ~concatenate_mrp
   ~mrp_prod_vector

   ~mrp_from_axis_angle
   ~mrp_from_quaternion

Jacobians
---------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~left_jacobian_SO3
   ~left_jacobian_SO3_series
   ~left_jacobian_SO3_inv
   ~left_jacobian_SO3_inv_series

Utility Functions
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~norm_angle
   ~norm_vector
   ~perpendicular_to_vectors
   ~angle_between_vectors
   ~vector_projection
   ~plane_basis_from_normal
   ~random_vector

Deprecated Functions
--------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~quaternion_from_extrinsic_euler_xyz
   ~active_matrix_from_intrinsic_euler_xzx
   ~active_matrix_from_extrinsic_euler_xzx
   ~active_matrix_from_intrinsic_euler_xyx
   ~active_matrix_from_extrinsic_euler_xyx
   ~active_matrix_from_intrinsic_euler_yxy
   ~active_matrix_from_extrinsic_euler_yxy
   ~active_matrix_from_intrinsic_euler_yzy
   ~active_matrix_from_extrinsic_euler_yzy
   ~active_matrix_from_intrinsic_euler_zyz
   ~active_matrix_from_extrinsic_euler_zyz
   ~active_matrix_from_intrinsic_euler_zxz
   ~active_matrix_from_extrinsic_euler_zxz
   ~active_matrix_from_intrinsic_euler_xzy
   ~active_matrix_from_extrinsic_euler_xzy
   ~active_matrix_from_intrinsic_euler_xyz
   ~active_matrix_from_extrinsic_euler_xyz
   ~active_matrix_from_intrinsic_euler_yxz
   ~active_matrix_from_extrinsic_euler_yxz
   ~active_matrix_from_intrinsic_euler_yzx
   ~active_matrix_from_extrinsic_euler_yzx
   ~active_matrix_from_intrinsic_euler_zyx
   ~active_matrix_from_extrinsic_euler_zyx
   ~active_matrix_from_intrinsic_euler_zxy
   ~active_matrix_from_extrinsic_euler_zxy
   ~active_matrix_from_extrinsic_roll_pitch_yaw
   ~intrinsic_euler_xzx_from_active_matrix
   ~extrinsic_euler_xzx_from_active_matrix
   ~intrinsic_euler_xyx_from_active_matrix
   ~extrinsic_euler_xyx_from_active_matrix
   ~intrinsic_euler_yxy_from_active_matrix
   ~extrinsic_euler_yxy_from_active_matrix
   ~intrinsic_euler_yzy_from_active_matrix
   ~extrinsic_euler_yzy_from_active_matrix
   ~intrinsic_euler_zyz_from_active_matrix
   ~extrinsic_euler_zyz_from_active_matrix
   ~intrinsic_euler_zxz_from_active_matrix
   ~extrinsic_euler_zxz_from_active_matrix
   ~intrinsic_euler_xzy_from_active_matrix
   ~extrinsic_euler_xzy_from_active_matrix
   ~intrinsic_euler_xyz_from_active_matrix
   ~extrinsic_euler_xyz_from_active_matrix
   ~intrinsic_euler_yxz_from_active_matrix
   ~extrinsic_euler_yxz_from_active_matrix
   ~intrinsic_euler_yzx_from_active_matrix
   ~extrinsic_euler_yzx_from_active_matrix
   ~intrinsic_euler_zyx_from_active_matrix
   ~extrinsic_euler_zyx_from_active_matrix
   ~intrinsic_euler_zxy_from_active_matrix
   ~extrinsic_euler_zxy_from_active_matrix


:mod:`pytransform3d.transformations`
====================================

.. automodule:: pytransform3d.transformations
    :no-members:
    :no-inherited-members:

Transformation Matrix
---------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_transform

   ~transform_requires_renormalization

   ~random_transform

   ~concat
   ~invert_transform
   ~transform
   ~vector_to_point
   ~vectors_to_points
   ~vector_to_direction
   ~vectors_to_directions
   ~adjoint_from_transform

   ~plot_transform

   ~assert_transform

   ~transform_from
   ~translate_transform
   ~rotate_transform

   ~transform_from_pq
   ~transform_from_exponential_coordinates
   ~transform_from_transform_log
   ~transform_from_dual_quaternion

Position and Quaternion
-----------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_pq

   ~pq_slerp

   ~pq_from_transform
   ~pq_from_dual_quaternion

Screw Parameters
----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_screw_parameters

   ~plot_screw

   ~assert_screw_parameters_equal

   ~screw_parameters_from_screw_axis
   ~screw_parameters_from_dual_quaternion

Screw Axis
----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_screw_axis

   ~random_screw_axis

   ~screw_axis_from_screw_parameters
   ~screw_axis_from_exponential_coordinates
   ~screw_axis_from_screw_matrix

Exponential Coordinates
-----------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_exponential_coordinates

   ~norm_exponential_coordinates

   ~random_exponential_coordinates

   ~assert_exponential_coordinates_equal

   ~exponential_coordinates_from_transform
   ~exponential_coordinates_from_screw_axis
   ~exponential_coordinates_from_transform_log

Screw Matrix
------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_screw_matrix

   ~screw_matrix_from_screw_axis
   ~screw_matrix_from_transform_log

Logarithm of Transformation
---------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_transform_log

   ~transform_log_from_exponential_coordinates
   ~transform_log_from_screw_matrix
   ~transform_log_from_transform

Dual Quaternion
---------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_dual_quaternion

   ~dual_quaternion_requires_renormalization

   ~assert_unit_dual_quaternion
   ~assert_unit_dual_quaternion_equal

   ~dual_quaternion_double

   ~dq_conj
   ~dq_q_conj
   ~concatenate_dual_quaternions
   ~dq_prod_vector
   ~dual_quaternion_power
   ~dual_quaternion_sclerp

   ~dual_quaternion_from_transform
   ~dual_quaternion_from_pq
   ~dual_quaternion_from_screw_parameters

Jacobians
---------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~left_jacobian_SE3
   ~left_jacobian_SE3_series
   ~left_jacobian_SE3_inv
   ~left_jacobian_SE3_inv_series

Deprecated Functions
--------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~scale_transform


:mod:`pytransform3d.batch_rotations`
====================================

.. automodule:: pytransform3d.batch_rotations
    :no-members:
    :no-inherited-members:

Rotation Matrices
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~active_matrices_from_angles

   ~active_matrices_from_intrinsic_euler_angles
   ~active_matrices_from_extrinsic_euler_angles
   ~matrices_from_compact_axis_angles
   ~matrices_from_quaternions

Axis-Angle Representation
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~norm_axis_angles
   ~axis_angles_from_matrices
   ~axis_angles_from_quaternions

   ~cross_product_matrices

Quaternions
-----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~batch_concatenate_quaternions
   ~batch_q_conj
   ~quaternion_slerp_batch
   ~smooth_quaternion_trajectory

   ~quaternions_from_matrices
   ~batch_quaternion_wxyz_from_xyzw
   ~batch_quaternion_xyzw_from_wxyz

Utility Functions
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~norm_vectors
   ~angles_between_vectors


:mod:`pytransform3d.trajectories`
=================================

.. automodule:: pytransform3d.trajectories
    :no-members:
    :no-inherited-members:

Transformation Matrices
-----------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~invert_transforms
   ~concat_one_to_many
   ~concat_many_to_one
   ~concat_many_to_many
   ~concat_dynamic

   ~transforms_from_pqs
   ~transforms_from_exponential_coordinates
   ~transforms_from_dual_quaternions

Positions and Quaternions
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~plot_trajectory

   ~pqs_from_transforms
   ~pqs_from_dual_quaternions

Screw Parameters
----------------
.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~screw_parameters_from_dual_quaternions

Exponential Coordinates
-----------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~mirror_screw_axis_direction

   ~exponential_coordinates_from_transforms

Dual Quaternions
----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~batch_dq_conj
   ~batch_dq_q_conj
   ~batch_concatenate_dual_quaternions
   ~batch_dq_prod_vector
   ~dual_quaternions_from_pqs
   ~dual_quaternions_power
   ~dual_quaternions_sclerp
   ~dual_quaternions_from_screw_parameters


:mod:`pytransform3d.uncertainty`
================================

.. automodule:: pytransform3d.uncertainty
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~estimate_gaussian_rotation_matrix_from_samples
   ~estimate_gaussian_transform_from_samples
   ~frechet_mean
   ~invert_uncertain_transform
   ~concat_globally_uncertain_transforms
   ~concat_locally_uncertain_transforms
   ~pose_fusion
   ~to_ellipsoid
   ~to_projected_ellipsoid
   ~plot_projected_ellipsoid


:mod:`pytransform3d.coordinates`
================================

.. automodule:: pytransform3d.coordinates
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~cartesian_from_cylindrical
   ~cartesian_from_spherical
   ~cylindrical_from_cartesian
   ~cylindrical_from_spherical
   ~spherical_from_cartesian
   ~spherical_from_cylindrical


:mod:`pytransform3d.transform_manager`
======================================

.. automodule:: pytransform3d.transform_manager
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~TransformGraphBase
   ~TransformManager
   ~TemporalTransformManager
   ~TimeVaryingTransform
   ~StaticTransform
   ~NumpyTimeseriesTransform


:mod:`pytransform3d.editor`
===========================

.. automodule:: pytransform3d.editor
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class_without_inherited.rst

   ~TransformEditor


:mod:`pytransform3d.urdf`
=========================

.. automodule:: pytransform3d.urdf
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~UrdfTransformManager
   ~Link
   ~Joint
   ~Geometry
   ~Box
   ~Sphere
   ~Cylinder
   ~Mesh

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~parse_urdf
   ~initialize_urdf_transform_manager


:mod:`pytransform3d.camera`
===========================

.. automodule:: pytransform3d.camera
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~make_world_grid
   ~make_world_line
   ~cam2sensor
   ~sensor2img
   ~world2image
   ~plot_camera


:mod:`pytransform3d.plot_utils`
===============================

.. automodule:: pytransform3d.plot_utils
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~make_3d_axis
   ~remove_frame
   ~plot_vector
   ~plot_length_variable
   ~plot_box
   ~plot_sphere
   ~plot_spheres
   ~plot_cylinder
   ~plot_mesh
   ~plot_ellipsoid
   ~plot_capsule
   ~plot_cone

.. autosummary::
   :toctree: _apidoc/
   :template: class_without_inherited.rst

   ~Arrow3D
   ~Frame
   ~LabeledFrame
   ~Trajectory
   ~Camera


:mod:`pytransform3d.visualizer`
===============================

.. automodule:: pytransform3d.visualizer
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~figure

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~Figure
   ~Artist
   ~Line3D
   ~PointCollection3D
   ~Vector3D
   ~Frame
   ~Trajectory
   ~Sphere
   ~Box
   ~Cylinder
   ~Mesh
   ~Ellipsoid
   ~Capsule
   ~Cone
   ~Plane
   ~Graph
   ~Camera
