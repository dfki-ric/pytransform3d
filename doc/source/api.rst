.. _api:

=================
API Documentation
=================

You can search for specific modules, classes or functions in the
:ref:`genindex`.


:mod:`pytransform3d.rotations`
============================

.. automodule:: pytransform3d.rotations
    :no-members:
    :no-inherited-members:

Utility Functions
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.norm_vector
   ~pytransform3d.rotations.norm_angle
   ~pytransform3d.rotations.norm_axis_angle
   ~pytransform3d.rotations.perpendicular_to_vectors
   ~pytransform3d.rotations.angle_between_vectors
   ~pytransform3d.rotations.random_vector
   ~pytransform3d.rotations.random_axis_angle
   ~pytransform3d.rotations.random_quaternion
   ~pytransform3d.rotations.cross_product_matrix

Input Validation Functions
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.check_matrix
   ~pytransform3d.rotations.check_axis_angle
   ~pytransform3d.rotations.check_quaternion

Conversions to Rotation Matrix
------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.matrix_from_axis_angle
   ~pytransform3d.rotations.matrix_from_quaternion
   ~pytransform3d.rotations.matrix_from_angle
   ~pytransform3d.rotations.matrix_from_euler_xyz
   ~pytransform3d.rotations.matrix_from_euler_zyx
   ~pytransform3d.rotations.matrix_from

Conversions to Euler Angles
---------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.euler_xyz_from_matrix
   ~pytransform3d.rotations.euler_zyx_from_matrix

Conversions to Axis-Angle
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.axis_angle_from_matrix
   ~pytransform3d.rotations.axis_angle_from_quaternion
   ~pytransform3d.rotations.compact_axis_angle

Conversions to Quaternion
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.quaternion_from_matrix
   ~pytransform3d.rotations.quaternion_from_axis_angle
   ~pytransform3d.rotations.quaternion_xyzw_from_wxyz
   ~pytransform3d.rotations.quaternion_wxyz_from_xyzw

Quaternion and Axis-Angle Operations
------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.concatenate_quaternions
   ~pytransform3d.rotations.q_prod_vector
   ~pytransform3d.rotations.q_conj
   ~pytransform3d.rotations.axis_angle_slerp
   ~pytransform3d.rotations.quaternion_slerp
   ~pytransform3d.rotations.quaternion_dist
   ~pytransform3d.rotations.quaternion_diff

Plotting
--------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.plot_basis
   ~pytransform3d.rotations.plot_axis_angle

Testing
-------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.rotations.assert_axis_angle_equal
   ~pytransform3d.rotations.assert_quaternion_equal
   ~pytransform3d.rotations.assert_euler_xyz_equal
   ~pytransform3d.rotations.assert_euler_zyx_equal
   ~pytransform3d.rotations.assert_rotation_matrix


:mod:`pytransform3d.transformations`
==================================

.. automodule:: pytransform3d.transformations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.transformations.check_transform
   ~pytransform3d.transformations.check_pq
   ~pytransform3d.transformations.transform_from
   ~pytransform3d.transformations.random_transform
   ~pytransform3d.transformations.invert_transform
   ~pytransform3d.transformations.translate_transform
   ~pytransform3d.transformations.rotate_transform
   ~pytransform3d.transformations.vector_to_point
   ~pytransform3d.transformations.concat
   ~pytransform3d.transformations.transform
   ~pytransform3d.transformations.scale_transform
   ~pytransform3d.transformations.plot_transform
   ~pytransform3d.transformations.assert_transform
   ~pytransform3d.transformations.pq_from_transform
   ~pytransform3d.transformations.transform_from_pq


:mod:`pytransform3d.trajectories`
===============================

.. automodule:: pytransform3d.trajectories
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.trajectories.matrices_from_pos_quat
   ~pytransform3d.trajectories.plot_trajectory


:mod:`pytransform3d.transform_manager`
====================================

.. automodule:: pytransform3d.transform_manager
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.transform_manager.TransformManager


:mod:`pytransform3d.editor`
=========================

.. automodule:: pytransform3d.editor
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.editor.TransformEditor


:mod:`pytransform3d.urdf`
=======================

.. automodule:: pytransform3d.urdf
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.urdf.UrdfTransformManager


:mod:`pytransform3d.camera`
=========================

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


:mod:`pytransform3d.plot_utils`
=============================

.. automodule:: pytransform3d.plot_utils
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform3d.plot_utils.make_3d_axis

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform3d.plot_utils.Arrow3D
   ~pytransform3d.plot_utils.Frame
   ~pytransform3d.plot_utils.Trajectory
