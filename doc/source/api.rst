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

Input Validation Functions
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_matrix
   ~check_skew_symmetric_matrix
   ~check_axis_angle
   ~check_compact_axis_angle
   ~check_quaternion
   ~check_quaternions
   ~check_rotor
   ~check_mrp

Conversions
-----------

Conversions to Rotation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See also :doc:`user_guide/euler_angles` for conversions from and to Euler
angles that have been omitted here for the sake of brevity.

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~passive_matrix_from_angle
   ~active_matrix_from_angle
   ~matrix_from_euler
   ~matrix_from_two_vectors
   ~matrix_from_axis_angle
   ~matrix_from_compact_axis_angle
   ~matrix_from_quaternion
   ~matrix_from_rotor

Conversions to Axis-Angle
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~axis_angle_from_matrix
   ~axis_angle_from_quaternion
   ~axis_angle_from_compact_axis_angle
   ~compact_axis_angle
   ~compact_axis_angle_from_matrix
   ~compact_axis_angle_from_quaternion

Conversions to Quaternion
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~quaternion_from_angle
   ~quaternion_from_matrix
   ~quaternion_from_axis_angle
   ~quaternion_from_compact_axis_angle
   ~quaternion_from_euler
   ~quaternion_from_mrp
   ~quaternion_xyzw_from_wxyz
   ~quaternion_wxyz_from_xyzw

Conversions to Rotor
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~rotor_from_two_directions
   ~rotor_from_plane_angle

Conversions to Modified Rodrigues Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~mrp_from_quaternion

Quaternion and Axis-Angle Operations
------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~axis_angle_from_two_directions
   ~axis_angle_slerp
   ~concatenate_quaternions
   ~q_prod_vector
   ~q_conj
   ~pick_closest_quaternion
   ~quaternion_slerp
   ~quaternion_dist
   ~quaternion_diff
   ~quaternion_gradient
   ~quaternion_integrate

Rotors
------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~wedge
   ~plane_normal_from_bivector
   ~geometric_product
   ~concatenate_rotors
   ~rotor_reverse
   ~rotor_apply
   ~rotor_slerp

Plotting
--------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~plot_basis
   ~plot_axis_angle
   ~plot_bivector

Testing
-------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~assert_axis_angle_equal
   ~assert_compact_axis_angle_equal
   ~assert_quaternion_equal
   ~assert_rotation_matrix

Normalization
-------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~norm_vector
   ~norm_matrix
   ~norm_angle
   ~norm_axis_angle
   ~norm_compact_axis_angle

Random Sampling
---------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~random_vector
   ~random_axis_angle
   ~random_compact_axis_angle
   ~random_quaternion

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

   ~perpendicular_to_vectors
   ~angle_between_vectors
   ~vector_projection
   ~plane_basis_from_normal
   ~cross_product_matrix

Deprecated Functions
--------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~quaternion_from_extrinsic_euler_xyz


:mod:`pytransform3d.transformations`
====================================

.. automodule:: pytransform3d.transformations
    :no-members:
    :no-inherited-members:

Input Validation Functions
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~check_transform
   ~check_pq
   ~check_screw_parameters
   ~check_screw_axis
   ~check_exponential_coordinates
   ~check_screw_matrix
   ~check_transform_log
   ~check_dual_quaternion

Conversions
-----------

Conversions to Transformation Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst


   ~transform_from
   ~translate_transform
   ~rotate_transform
   ~transform_from_pq
   ~transform_from_exponential_coordinates
   ~transform_from_transform_log
   ~transform_from_dual_quaternion

Conversions to Position and Quaternion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pq_from_transform
   ~pq_from_dual_quaternion

Conversions to Screw Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~screw_parameters_from_screw_axis
   ~screw_parameters_from_dual_quaternion

Conversions to Screw Axis
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~screw_axis_from_screw_parameters
   ~screw_axis_from_exponential_coordinates
   ~screw_axis_from_screw_matrix

Conversions to Exponential Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~exponential_coordinates_from_transform
   ~exponential_coordinates_from_screw_axis
   ~exponential_coordinates_from_transform_log

Conversions to Screw Matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~screw_matrix_from_screw_axis
   ~screw_matrix_from_transform_log

Conversions to Matrix Logarithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~transform_log_from_exponential_coordinates
   ~transform_log_from_screw_matrix
   ~transform_log_from_transform

Conversions to Dual Quaternions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~dual_quaternion_from_transform
   ~dual_quaternion_from_pq
   ~dual_quaternion_from_screw_parameters

Apply Transformations
---------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~concat
   ~invert_transform
   ~transform
   ~vector_to_point
   ~vectors_to_points
   ~vector_to_direction
   ~vectors_to_directions
   ~scale_transform
   ~adjoint_from_transform

Position+Quaternion Operations
------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pq_slerp

Dual Quaternion Operations
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~dq_conj
   ~dq_q_conj
   ~concatenate_dual_quaternions
   ~dq_prod_vector
   ~dual_quaternion_power
   ~dual_quaternion_sclerp

Plotting
--------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~plot_transform
   ~plot_screw

Testing
-------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~assert_transform
   ~assert_unit_dual_quaternion
   ~assert_unit_dual_quaternion_equal
   ~assert_screw_parameters_equal

Random Sampling
---------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~random_transform
   ~random_screw_axis
   ~random_exponential_coordinates

Normalization
-------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~norm_exponential_coordinates

Jacobians
---------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~left_jacobian_SE3
   ~left_jacobian_SE3_series
   ~left_jacobian_SE3_inv
   ~left_jacobian_SE3_inv_series


:mod:`pytransform3d.batch_rotations`
====================================

.. automodule:: pytransform3d.batch_rotations
    :no-members:
    :no-inherited-members:

Conversions
-----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~active_matrices_from_angles
   ~active_matrices_from_intrinsic_euler_angles
   ~active_matrices_from_extrinsic_euler_angles
   ~matrices_from_compact_axis_angles
   ~axis_angles_from_matrices
   ~cross_product_matrices
   ~matrices_from_quaternions
   ~quaternions_from_matrices
   ~batch_quaternion_wxyz_from_xyzw
   ~batch_quaternion_xyzw_from_wxyz

Operations
----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~batch_q_conj
   ~batch_concatenate_quaternions
   ~quaternion_slerp_batch
   ~smooth_quaternion_trajectory

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

Conversions
-----------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~transforms_from_pqs
   ~transforms_from_exponential_coordinates
   ~transforms_from_dual_quaternions
   ~pqs_from_transforms
   ~pqs_from_dual_quaternions
   ~exponential_coordinates_from_transforms
   ~dual_quaternions_from_pqs

Operations and Utility Functions
--------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~invert_transforms
   ~concat_one_to_many
   ~concat_many_to_one
   ~batch_dq_conj
   ~batch_concatenate_dual_quaternions
   ~batch_dq_prod_vector
   ~plot_trajectory
   ~mirror_screw_axis_direction


:mod:`pytransform3d.uncertainty`
================================

.. automodule:: pytransform3d.uncertainty
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~estimate_gaussian_transform_from_samples
   ~invert_uncertain_transform
   ~concat_globally_uncertain_transforms
   ~concat_locally_uncertain_transforms
   ~pose_fusion
   ~to_ellipsoid
   ~to_projected_ellipsoid
   ~plot_projected_ellipsoid


:mod:`pytransform3d.coordinates`
=================================

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
