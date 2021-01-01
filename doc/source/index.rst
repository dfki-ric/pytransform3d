.. pytransform3d documentation master file, created by
   sphinx-quickstart on Thu Nov 20 21:01:30 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
pytransform3d
=============

pytransform3d covers the following groups of transformations.

+-------+--------------------+-----------------------------------------+
| Group | Description        | Representations                         |
+=======+====================+=========================================+
| SO(3) | 3D rotations       | unit quaternion, rotation matrix,       |
|       |                    | axis-angle, Euler angles                |
+-------+--------------------+-----------------------------------------+
| SE(3) | 3D transformations | transformation matrix, translation      |
|       | (rotation and      | and unit quaternion, exponential        |
|       | translation)       | coordinates, logarithm of               |
|       |                    | transformation                          |
+-------+--------------------+-----------------------------------------+

------------
Installation
------------

pytransform3d is available at
`GitHub <https://github.com/rock-learning/pytransform3d>`_.
The readme there contains installation instructions. This documentation
explains how you can work with pytransform3d and with 3D transformations
in general.

-----------------
Table of Contents
-----------------

.. toctree::
   :maxdepth: 1

   notation
   rotations
   transformations
   transformation_ambiguities
   euler_angles
   transformation_modeling
   transform_manager
   camera
   animations
   api
   _auto_examples/index
