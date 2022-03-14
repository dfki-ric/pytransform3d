.. _api:

=================
API Documentation
=================

This is the detailed documentation of all public classes and functions.
You can also search for specific modules, classes, or functions in the
:ref:`genindex`.


.. contents:: :local:
    :depth: 1


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

   ~pytransform3d.rotations.check_matrix
   ~pytransform3d.rotations.check_skew_symmetric_matrix
   ~pytransform3d.rotations.check_axis_angle
   ~pytransform3d.rotations.check_compact_axis_angle
   ~pytransform3d.rotations.check_quaternion
   ~pytransform3d.rotations.check_quaternions
   ~pytransform3d.rotations.check_rotor

Conversions to Rotation Matrix
------------------------------

See also :doc:`euler_angles` for conversions from and to Euler angles
that have been omitted here for the sake of brevity.

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.passive_matrix_from_angle
   ~pytransform3d.rotations.active_matrix_from_angle
   ~pytransform3d.rotations.matrix_from_two_vectors
   ~pytransform3d.rotations.matrix_from_axis_angle
   ~pytransform3d.rotations.matrix_from_compact_axis_angle
   ~pytransform3d.rotations.matrix_from_quaternion
   ~pytransform3d.rotations.matrix_from_rotor

Conversions to Axis-Angle
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.axis_angle_from_matrix
   ~pytransform3d.rotations.axis_angle_from_quaternion
   ~pytransform3d.rotations.axis_angle_from_compact_axis_angle
   ~pytransform3d.rotations.compact_axis_angle
   ~pytransform3d.rotations.compact_axis_angle_from_matrix
   ~pytransform3d.rotations.compact_axis_angle_from_quaternion

Conversions to Quaternion
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.quaternion_from_matrix
   ~pytransform3d.rotations.quaternion_from_axis_angle
   ~pytransform3d.rotations.quaternion_from_compact_axis_angle
   ~pytransform3d.rotations.quaternion_xyzw_from_wxyz
   ~pytransform3d.rotations.quaternion_wxyz_from_xyzw

Conversions to Rotor
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.rotor_from_two_directions
   ~pytransform3d.rotations.rotor_from_plane_angle

Quaternion and Axis-Angle Operations
------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.axis_angle_from_two_directions
   ~pytransform3d.rotations.axis_angle_slerp
   ~pytransform3d.rotations.concatenate_quaternions
   ~pytransform3d.rotations.q_prod_vector
   ~pytransform3d.rotations.q_conj
   ~pytransform3d.rotations.quaternion_slerp
   ~pytransform3d.rotations.quaternion_dist
   ~pytransform3d.rotations.quaternion_diff
   ~pytransform3d.rotations.quaternion_gradient
   ~pytransform3d.rotations.quaternion_integrate

Rotors
------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.wedge
   ~pytransform3d.rotations.plane_normal_from_bivector
   ~pytransform3d.rotations.geometric_product
   ~pytransform3d.rotations.concatenate_rotors
   ~pytransform3d.rotations.rotor_reverse
   ~pytransform3d.rotations.rotor_apply
   ~pytransform3d.rotations.rotor_slerp

Plotting
--------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.plot_basis
   ~pytransform3d.rotations.plot_axis_angle
   ~pytransform3d.rotations.plot_bivector

Testing
-------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.assert_axis_angle_equal
   ~pytransform3d.rotations.assert_compact_axis_angle_equal
   ~pytransform3d.rotations.assert_quaternion_equal
   ~pytransform3d.rotations.assert_rotation_matrix

Utility Functions
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.norm_vector
   ~pytransform3d.rotations.norm_matrix
   ~pytransform3d.rotations.norm_angle
   ~pytransform3d.rotations.norm_axis_angle
   ~pytransform3d.rotations.norm_compact_axis_angle
   ~pytransform3d.rotations.perpendicular_to_vectors
   ~pytransform3d.rotations.angle_between_vectors
   ~pytransform3d.rotations.vector_projection
   ~pytransform3d.rotations.plane_basis_from_normal
   ~pytransform3d.rotations.random_vector
   ~pytransform3d.rotations.random_axis_angle
   ~pytransform3d.rotations.random_compact_axis_angle
   ~pytransform3d.rotations.random_quaternion
   ~pytransform3d.rotations.cross_product_matrix


Deprecated Functions
--------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.matrix_from
   ~pytransform3d.rotations.matrix_from_angle
   ~pytransform3d.rotations.matrix_from_euler_xyz
   ~pytransform3d.rotations.matrix_from_euler_zyx
   ~pytransform3d.rotations.euler_xyz_from_matrix
   ~pytransform3d.rotations.euler_zyx_from_matrix
   ~pytransform3d.rotations.assert_euler_xyz_equal
   ~pytransform3d.rotations.assert_euler_zyx_equal


:mod:`pytransform3d.transformations`
====================================

.. automodule:: pytransform3d.transformations
    :no-members:
    :no-inherited-members:

Utility Functions
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.random_transform
   ~pytransform3d.transformations.random_screw_axis
   ~pytransform3d.transformations.norm_exponential_coordinates

Input Validation Functions
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.check_transform
   ~pytransform3d.transformations.check_pq
   ~pytransform3d.transformations.check_screw_parameters
   ~pytransform3d.transformations.check_screw_axis
   ~pytransform3d.transformations.check_exponential_coordinates
   ~pytransform3d.transformations.check_screw_matrix
   ~pytransform3d.transformations.check_transform_log
   ~pytransform3d.transformations.check_dual_quaternion

Conversions to Transformation Matrix
------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst


   ~pytransform3d.transformations.transform_from
   ~pytransform3d.transformations.translate_transform
   ~pytransform3d.transformations.rotate_transform
   ~pytransform3d.transformations.transform_from_pq
   ~pytransform3d.transformations.transform_from_exponential_coordinates
   ~pytransform3d.transformations.transform_from_transform_log
   ~pytransform3d.transformations.transform_from_dual_quaternion

Conversions to Position and Quaternion
--------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.pq_from_transform
   ~pytransform3d.transformations.pq_from_dual_quaternion

Conversions to Screw Parameters
-------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.screw_parameters_from_screw_axis
   ~pytransform3d.transformations.screw_parameters_from_dual_quaternion

Conversions to Screw Axis
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.screw_axis_from_screw_parameters
   ~pytransform3d.transformations.screw_axis_from_exponential_coordinates
   ~pytransform3d.transformations.screw_axis_from_screw_matrix

Conversions to Exponential Coordinates
--------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.exponential_coordinates_from_transform
   ~pytransform3d.transformations.exponential_coordinates_from_screw_axis
   ~pytransform3d.transformations.exponential_coordinates_from_transform_log

Conversions to Screw Matrix
---------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.screw_matrix_from_screw_axis
   ~pytransform3d.transformations.screw_matrix_from_transform_log

Conversions to Matrix Logarithm
-------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.transform_log_from_exponential_coordinates
   ~pytransform3d.transformations.transform_log_from_screw_matrix
   ~pytransform3d.transformations.transform_log_from_transform

Conversions to Dual Quaternions
-------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.dual_quaternion_from_transform
   ~pytransform3d.transformations.dual_quaternion_from_pq
   ~pytransform3d.transformations.dual_quaternion_from_screw_parameters

Apply Transformations
---------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.concat
   ~pytransform3d.transformations.invert_transform
   ~pytransform3d.transformations.transform
   ~pytransform3d.transformations.vector_to_point
   ~pytransform3d.transformations.vectors_to_points
   ~pytransform3d.transformations.vector_to_direction
   ~pytransform3d.transformations.vectors_to_directions
   ~pytransform3d.transformations.scale_transform
   ~pytransform3d.transformations.adjoint_from_transform

Dual Quaternion Operations
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.dq_conj
   ~pytransform3d.transformations.dq_q_conj
   ~pytransform3d.transformations.concatenate_dual_quaternions
   ~pytransform3d.transformations.dq_prod_vector
   ~pytransform3d.transformations.dual_quaternion_power
   ~pytransform3d.transformations.dual_quaternion_sclerp

Plotting
--------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.plot_transform
   ~pytransform3d.transformations.plot_screw

Testing
-------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.assert_transform
   ~pytransform3d.transformations.assert_unit_dual_quaternion
   ~pytransform3d.transformations.assert_unit_dual_quaternion_equal
   ~pytransform3d.transformations.assert_screw_parameters_equal


:mod:`pytransform3d.batch_rotations`
====================================

.. automodule:: pytransform3d.batch_rotations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.batch_rotations.norm_vectors
   ~pytransform3d.batch_rotations.angles_between_vectors
   ~pytransform3d.batch_rotations.active_matrices_from_angles
   ~pytransform3d.batch_rotations.active_matrices_from_intrinsic_euler_angles
   ~pytransform3d.batch_rotations.active_matrices_from_extrinsic_euler_angles
   ~pytransform3d.batch_rotations.matrices_from_compact_axis_angles
   ~pytransform3d.batch_rotations.axis_angles_from_matrices
   ~pytransform3d.batch_rotations.cross_product_matrices
   ~pytransform3d.batch_rotations.matrices_from_quaternions
   ~pytransform3d.batch_rotations.quaternions_from_matrices
   ~pytransform3d.batch_rotations.quaternion_slerp_batch
   ~pytransform3d.batch_rotations.batch_q_conj
   ~pytransform3d.batch_rotations.batch_concatenate_quaternions
   ~pytransform3d.batch_rotations.batch_quaternion_wxyz_from_xyzw
   ~pytransform3d.batch_rotations.batch_quaternion_xyzw_from_wxyz
   ~pytransform3d.batch_rotations.smooth_quaternion_trajectory


:mod:`pytransform3d.trajectories`
=================================

.. automodule:: pytransform3d.trajectories
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.trajectories.invert_transforms
   ~pytransform3d.trajectories.concat_one_to_many
   ~pytransform3d.trajectories.concat_many_to_one
   ~pytransform3d.trajectories.transforms_from_pqs
   ~pytransform3d.trajectories.pqs_from_transforms
   ~pytransform3d.trajectories.exponential_coordinates_from_transforms
   ~pytransform3d.trajectories.transforms_from_exponential_coordinates
   ~pytransform3d.trajectories.dual_quaternions_from_pqs
   ~pytransform3d.trajectories.pqs_from_dual_quaternions
   ~pytransform3d.trajectories.batch_dq_conj
   ~pytransform3d.trajectories.batch_concatenate_dual_quaternions
   ~pytransform3d.trajectories.batch_dq_prod_vector
   ~pytransform3d.trajectories.plot_trajectory


:mod:`pytransform3d.coordinates`
=================================

.. automodule:: pytransform3d.coordinates
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.coordinates.cartesian_from_cylindrical
   ~pytransform3d.coordinates.cartesian_from_spherical
   ~pytransform3d.coordinates.cylindrical_from_cartesian
   ~pytransform3d.coordinates.cylindrical_from_spherical
   ~pytransform3d.coordinates.spherical_from_cartesian
   ~pytransform3d.coordinates.spherical_from_cylindrical


:mod:`pytransform3d.transform_manager`
======================================

.. automodule:: pytransform3d.transform_manager
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.transform_manager.TransformManager


:mod:`pytransform3d.editor`
===========================

.. automodule:: pytransform3d.editor
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class_without_inherited.rst

   ~pytransform3d.editor.TransformEditor


:mod:`pytransform3d.urdf`
=========================

.. automodule:: pytransform3d.urdf
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.urdf.UrdfTransformManager
   ~pytransform3d.urdf.Link
   ~pytransform3d.urdf.Joint
   ~pytransform3d.urdf.Geometry
   ~pytransform3d.urdf.Box
   ~pytransform3d.urdf.Sphere
   ~pytransform3d.urdf.Cylinder
   ~pytransform3d.urdf.Mesh

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.urdf.parse_urdf
   ~pytransform3d.urdf.initialize_urdf_transform_manager


:mod:`pytransform3d.camera`
===========================

.. automodule:: pytransform3d.camera
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.camera.make_world_grid
   ~pytransform3d.camera.make_world_line
   ~pytransform3d.camera.cam2sensor
   ~pytransform3d.camera.sensor2img
   ~pytransform3d.camera.world2image
   ~pytransform3d.camera.plot_camera


:mod:`pytransform3d.plot_utils`
===============================

.. automodule:: pytransform3d.plot_utils
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.plot_utils.make_3d_axis
   ~pytransform3d.plot_utils.remove_frame
   ~pytransform3d.plot_utils.plot_vector
   ~pytransform3d.plot_utils.plot_length_variable
   ~pytransform3d.plot_utils.plot_box
   ~pytransform3d.plot_utils.plot_sphere
   ~pytransform3d.plot_utils.plot_cylinder
   ~pytransform3d.plot_utils.plot_mesh
   ~pytransform3d.plot_utils.plot_ellipsoid
   ~pytransform3d.plot_utils.plot_capsule
   ~pytransform3d.plot_utils.plot_cone

.. autosummary::
   :toctree: _apidoc/
   :template: class_without_inherited.rst

   ~pytransform3d.plot_utils.Arrow3D
   ~pytransform3d.plot_utils.Frame
   ~pytransform3d.plot_utils.LabeledFrame
   ~pytransform3d.plot_utils.Trajectory


:mod:`pytransform3d.visualizer`
===============================

.. automodule:: pytransform3d.visualizer
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.visualizer.figure

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.visualizer.Figure
   ~pytransform3d.visualizer.Artist
   ~pytransform3d.visualizer.Line3D
   ~pytransform3d.visualizer.PointCollection3D
   ~pytransform3d.visualizer.Vector3D
   ~pytransform3d.visualizer.Frame
   ~pytransform3d.visualizer.Trajectory
   ~pytransform3d.visualizer.Sphere
   ~pytransform3d.visualizer.Box
   ~pytransform3d.visualizer.Cylinder
   ~pytransform3d.visualizer.Mesh
   ~pytransform3d.visualizer.Ellipsoid
   ~pytransform3d.visualizer.Capsule
   ~pytransform3d.visualizer.Cone
   ~pytransform3d.visualizer.Graph
