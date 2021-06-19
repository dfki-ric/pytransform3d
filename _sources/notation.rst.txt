========
Notation
========

For physical quantities we use the notation :math:`_{A}\boldsymbol{x}_{BC}`,
where :math:`\boldsymbol{x}` is a physical quantity of frame C with
respect to frame B expressed in frame A, where frame refers to a reference
frame or coordinate system that is defined by three orthonormal basis vectors
and a position in three-dimensional space. For example,
:math:`_{A}\boldsymbol{t}_{BC}` is the translation of C with respect to B
measured in A or :math:`_{A}\boldsymbol{\omega}_{BC}` is the
orientation vector of C with respect to B measured in A.

Since :math:`_A\boldsymbol{t}_{BC}` represents a vector or translation from
frame B to frame C expressed in frame A, the position of a point :math:`P`
with respect to a frame A in three-dimensional space can be defined by
:math:`_A\boldsymbol{p} := _A\boldsymbol{t}_{AP}`.

When we define a mapping from some frame A to another frame B that can be
expressed as a matrix multiplication, we use the notation
:math:`\boldsymbol{M}_{BA}` for the corresponding matrix. We can read this
from right to left as a matrix that maps from frame A to frame B through
multiplication, for example, when we want to transform a point by

.. math::

    _B\boldsymbol{p} = \boldsymbol{M}_{BA} {_A\boldsymbol{p}}

---------------
Representations
---------------

We can use many different representations of rotation and / or translation.
Here is an overview of the representations that are available in pytransform3d.
All representations are stored in NumPy arrays, of which the corresponding
shape is shown in this table. You will find more details on these
representations on the following pages.

+----------------------------------------+---------------------+----------+-------------+
| Representation and Mathematical Symbol | NumPy Array Shape   | Rotation | Translation |
+========================================+=====================+==========+=============+
| Rotation matrix                        | (3, 3)              | X        |             |
| :math:`\pmb{R}`                        |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Compact axis-angle                     | (3,)                | X        |             |
| :math:`\pmb{\omega}`                   |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Axis-angle                             | (4,)                | X        |             |
| :math:`(\hat{\pmb{\omega}}, \theta)`   |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Logarithm of rotation                  | (3, 3)              | X        |             |
| :math:`\left[\pmb{\omega}\right]`      |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Quaternion                             | (4,)                | X        |             |
| :math:`\pmb{q}`                        |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Rotor                                  | (4,)                | X        |             |
| :math:`R`                              |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Euler angles                           | (3,)                | X        |             |
| :math:`(\alpha, \beta, \gamma)`        |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Transformation matrix                  | (4, 4)              | X        | X           |
| :math:`\pmb{T}`                        |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Exponential coordinates                | (6,)                | X        | X           |
| :math:`\mathcal{S}\theta`              |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Logarithm of transformation            | (4, 4)              | X        | X           |
| :math:`\left[\mathcal{S}\right]\theta` |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Position and quaternion                | (7,)                | X        | X           |
| :math:`(\pmb{p}, \pmb{q})`             |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+
| Dual quaternion                        | (8,)                | X        | X           |
| :math:`\pmb{p} + \epsilon\pmb{q}`      |                     |          |             |
+----------------------------------------+---------------------+----------+-------------+

------------------------------------
Duality of Transformations and Poses
------------------------------------

We can use a transformation matrix :math:`\boldsymbol{T}_{BA}` that represents
a transformation from frame A to frame B to represent the pose (position and
orientation) of frame A in frame B (if we use the active transformation
convention; see :ref:`transformation_ambiguities` for details). This is just
a different interpretation of the same matrix and similar to our interpretation
of a vector from A to P :math:`_A\boldsymbol{t}_{AP}` as a point
:math:`_A\boldsymbol{p}`.

----------
References
----------

* Representing Robot Pose: The good, the bad, and the ugly (slides): http://static.squarespace.com/static/523c5c56e4b0abc2df5e163e/t/53957839e4b05045ad65021d/1402304569659/Workshop+-+Rotations_v102.key.pdf
* Representing Robot Pose: The good, the bad, and the ugly (blog): http://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly
