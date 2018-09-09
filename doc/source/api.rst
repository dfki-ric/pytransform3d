.. _api:

=================
API Documentation
=================

You can search for specific modules, classes or functions in the
:ref:`genindex`.


:mod:`pytransform.rotations`
============================

.. automodule:: pytransform.rotations
    :no-members:
    :no-inherited-members:

Utility Functions
-----------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.norm_vector
   ~pytransform.rotations.norm_angle
   ~pytransform.rotations.norm_axis_angle
   ~pytransform.rotations.perpendicular_to_vectors
   ~pytransform.rotations.angle_between_vectors
   ~pytransform.rotations.random_vector
   ~pytransform.rotations.random_axis_angle
   ~pytransform.rotations.random_quaternion
   ~pytransform.rotations.cross_product_matrix

Input Validation Functions
--------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.check_matrix
   ~pytransform.rotations.check_axis_angle
   ~pytransform.rotations.check_quaternion

Conversions to Rotation Matrix
------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.matrix_from_axis_angle
   ~pytransform.rotations.matrix_from_quaternion
   ~pytransform.rotations.matrix_from_angle
   ~pytransform.rotations.matrix_from_euler_xyz
   ~pytransform.rotations.matrix_from_euler_zyx
   ~pytransform.rotations.matrix_from

Conversions to Euler Angles
---------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.euler_xyz_from_matrix
   ~pytransform.rotations.euler_zyx_from_matrix

Conversions to Axis-Angle
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.axis_angle_from_matrix
   ~pytransform.rotations.axis_angle_from_quaternion
   ~pytransform.rotations.compact_axis_angle

Conversions to Quaternion
-------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.quaternion_from_matrix
   ~pytransform.rotations.quaternion_from_axis_angle
   ~pytransform.rotations.quaternion_xyzw_from_wxyz
   ~pytransform.rotations.quaternion_wxyz_from_xyzw

Quaternion and Axis-Angle Operations
------------------------------------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.concatenate_quaternions
   ~pytransform.rotations.q_prod_vector
   ~pytransform.rotations.q_conj
   ~pytransform.rotations.axis_angle_slerp
   ~pytransform.rotations.quaternion_slerp
   ~pytransform.rotations.quaternion_dist
   ~pytransform.rotations.quaternion_diff

Plotting
--------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.plot_basis
   ~pytransform.rotations.plot_axis_angle

Testing
-------

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.assert_axis_angle_equal
   ~pytransform.rotations.assert_quaternion_equal
   ~pytransform.rotations.assert_euler_xyz_equal
   ~pytransform.rotations.assert_euler_zyx_equal
   ~pytransform.rotations.assert_rotation_matrix


:mod:`pytransform.transformations`
==================================

.. automodule:: pytransform.transformations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.transformations.check_transform
   ~pytransform.transformations.check_pq
   ~pytransform.transformations.transform_from
   ~pytransform.transformations.random_transform
   ~pytransform.transformations.invert_transform
   ~pytransform.transformations.translate_transform
   ~pytransform.transformations.rotate_transform
   ~pytransform.transformations.vector_to_point
   ~pytransform.transformations.concat
   ~pytransform.transformations.transform
   ~pytransform.transformations.scale_transform
   ~pytransform.transformations.plot_transform
   ~pytransform.transformations.assert_transform
   ~pytransform.transformations.pq_from_transform
   ~pytransform.transformations.transform_from_pq


:mod:`pytransform.trajectories`
===============================

.. automodule:: pytransform.trajectories
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.trajectories.matrices_from_pos_quat
   ~pytransform.trajectories.plot_trajectory


:mod:`pytransform.transform_manager`
====================================

.. automodule:: pytransform.transform_manager
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform.transform_manager.TransformManager


:mod:`pytransform.editor`
=========================

.. automodule:: pytransform.editor
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform.editor.TransformEditor


:mod:`pytransform.urdf`
=======================

.. automodule:: pytransform.urdf
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~pytransform.urdf.UrdfTransformManager


:mod:`pytransform.camera`
=========================

.. automodule:: pytransform.camera
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.camera.make_world_grid
   ~pytransform.camera.make_world_line
   ~pytransform.camera.cam2sensor
   ~pytransform.camera.sensor2img
   ~pytransform.camera.world2image
