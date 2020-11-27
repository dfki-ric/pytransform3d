.. pytransform3d documentation master file, created by
   sphinx-quickstart on Thu Nov 20 21:01:30 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
pytransform3d
=============

pytransform3d covers the following groups of transformations.

+-------+-----------------+-----------+---------------------+
| Group | Description     | Dimension | Representation      |
+=======+=================+===========+=====================+
| SO(3) | 3D rotations    | 3         | unit quaternion,    |
|       |                 |           | rotation matrix,    |
|       |                 |           | axis-angle,         |
|       |                 |           | Euler angles        |
+-------+-----------------+-----------+---------------------+
| SE(3) | 3D rigid        | 6         | transformation      |
|       | transformations |           | matrix, translation |
|       | (translation +  |           | + unit quaternion   |
|       | rotation)       |           |                     |
+-------+-----------------+-----------+---------------------+

In this documentation we will use the notation :math:`_A\boldsymbol{t}_{BC}`
to represent a vector from frame B to frame C expressed in frame A.

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

----------
References
----------

* Representing Robot Pose: The good, the bad, and the ugly (slides): http://static.squarespace.com/static/523c5c56e4b0abc2df5e163e/t/53957839e4b05045ad65021d/1402304569659/Workshop+-+Rotations_v102.key.pdf 
* Representing Robot Pose: The good, the bad, and the ugly (blog): http://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly
* Kindr cheat sheet: https://docs.leggedrobotics.com/kindr/cheatsheet_latest.pdf
* Why and How to Avoid the Flipped Quaternion Multiplication: https://arxiv.org/pdf/1801.07478.pdf
