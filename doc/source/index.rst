.. pytransform3d documentation master file, created by
   sphinx-quickstart on Thu Nov 20 21:01:30 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
pytransform3d
=============

pytransform3d covers the following groups of transformations.

+-------+------------------+-----------+------------------------------------+
| Group | Description      | Dimension | Representation                     |
+=======+==================+===========+====================================+
| SO(3) | 3D rotations     | 3         | unit quaternion, rotation matrix,  |
|       |                  |           | axis-angle, Euler angles           |
+-------+------------------+-----------+------------------------------------+
| SE(3) | 3D rigid body    | 6         | transformation matrix, translation |
|       | transformations  |           | and unit quaternion, exponential   |
|       | (translation and |           | coordinates, logarithm of          |
|       | rotation)        |           | transformation                     |
+-------+------------------+-----------+------------------------------------+

In this documentation we will use the notation :math:`_A\boldsymbol{t}_{BC}`
to represent a vector from frame B to frame C expressed in frame A.

------------
Installation
------------

pytransform3d is available at GitHub: https://github.com/rock-learning/pytransform3d

The readme contains detailed installation instructions. This documentation
contains guidelines on working with pytransform3d and working with
transformations in general as well as an API documentation.

-----------------
Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   rotations
   transformations
   transformation_ambiguities
   transformation_modeling
   transform_manager
   camera
   animations
   api
