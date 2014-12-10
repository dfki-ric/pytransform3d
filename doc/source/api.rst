.. _api:

===
API
===

You can search for specific modules, classes or functions in the
:ref:`genindex`.


:mod:`pytransform.rotations`: Rotations
=======================================

.. automodule:: pytransform.rotations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.rotations.norm_vector
   ~pytransform.rotations.perpendicular_to_vectors
   ~pytransform.rotations.angle_between_vectors
   ~pytransform.rotations.random_vector
   ~pytransform.rotations.random_axis_angle
   ~pytransform.rotations.random_quaternion
   ~pytransform.rotations.cross_product_matrix
   ~pytransform.rotations.matrix_from_axis_angle
   ~pytransform.rotations.matrix_from_quaternion
   ~pytransform.rotations.matrix_from_angle
   ~pytransform.rotations.matrix_from_euler_xyz
   ~pytransform.rotations.matrix_from_euler_zyx
   ~pytransform.rotations.matrix_from
   ~pytransform.rotations.axis_angle_from_matrix
   ~pytransform.rotations.axis_angle_from_quaternion
   ~pytransform.rotations.quaternion_from_matrix
   ~pytransform.rotations.quaternion_from_axis_angle
   ~pytransform.rotations.concatenate_quaternions
   ~pytransform.rotations.q_prod_vector
   ~pytransform.rotations.q_conj
   ~pytransform.rotations.axis_angle_slerp
   ~pytransform.rotations.quaternion_slerp
   ~pytransform.rotations.quaternion_dist
   ~pytransform.rotations.plot_basis
   ~pytransform.rotations.plot_axis_angle
   ~pytransform.rotations.assert_quaternion_equal
   ~pytransform.rotations.assert_rotation_matrix


:mod:`pytransform.transformations`: Transformations
===================================================

.. automodule:: pytransform.transformations
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.transformations.invert_transform
   ~pytransform.transformations.transform_from
   ~pytransform.transformations.translate_transform
   ~pytransform.transformations.rotate_transform
   ~pytransform.transformations.vector_to_point
   ~pytransform.transformations.transform
   ~pytransform.transformations.plot_transform


:mod:`pytransform.camera`: Camera
=================================

.. automodule:: pytransform.camera
    :no-members:
    :no-inherited-members:

.. autosummary::
   :toctree: _apidoc/
   :template: function.rst

   ~pytransform.camera.make_world_grid
   ~pytransform.camera.cam2sensor
   ~pytransform.camera.sensor2img
   ~pytransform.camera.world2image
