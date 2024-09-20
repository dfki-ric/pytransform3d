=========================
SE(3): 3D Transformations
=========================

The group of all proper rigid transformations (rototranslations) in
3D Cartesian space is :math:`SE(3)` (SE: special Euclidean group).
Transformations consist of a rotation and a translation. Those can be
represented in different ways just like rotations can be expressed
in different ways.

For most representations of orientations we can find
an analogous representation of transformations [1]_:

* A **transformation matrix** :math:`\boldsymbol T` is similar to a rotation
  matrix :math:`\boldsymbol R`.
* A **screw axis** :math:`\mathcal S` is similar to a rotation axis
  :math:`\hat{\boldsymbol{\omega}}`.
* A **screw matrix** :math:`\left[\mathcal{S}\right]` is similar to
  a cross-product matrix of a unit rotation axis
  :math:`\left[\hat{\boldsymbol{\omega}}\right]`.
* The **logarithm of a transformation** :math:`\left[\mathcal{S}\right] \theta`
  is similar to a cross-product matrix of the angle-axis representation
  :math:`\left[\hat{\boldsymbol{\omega}}\right] \theta`.
* The **exponential coordinates** :math:`\mathcal{S} \theta` for rigid body
  motions are similar to exponential coordinates
  :math:`\hat{\boldsymbol{\omega}} \theta` for rotations (compact axis-angle
  representation / rotation vector).
* A **twist** :math:`\mathcal V = \mathcal{S} \dot{\theta}` is similar to
  angular velocity :math:`\hat{\boldsymbol{\omega}} \dot{\theta}`.
* A (unit) **dual quaternion**
  :math:`p_w + p_x i + p_y j + p_z k + \epsilon (q_w + q_x i + q_y j + q_z k)`
  is similar to a (unit) quaternion :math:`w + x i + y j + z k`.

Here is an overview of the representations and the conversions between them
that are available in pytransform3d.

.. image:: ../_static/transformations.png
   :alt: Transformations
   :width: 50%
   :align: center


---------------------
Transformation Matrix
---------------------

One of the most convenient ways to represent transformations are
transformation matrices. A transformation matrix is a 4x4 matrix of
the form

.. math::

    \boldsymbol T =
    \left( \begin{array}{cc}
        \boldsymbol R & \boldsymbol t\\
        \boldsymbol 0 & 1\\
    \end{array} \right)
    =
    \left(
    \begin{matrix}
    r_{11} & r_{12} & r_{13} & t_1\\
    r_{21} & r_{22} & r_{23} & t_2\\
    r_{31} & r_{32} & r_{33} & t_3\\
    0 & 0 & 0 & 1\\
    \end{matrix}
    \right)
    \in SE(3).

It is a partitioned matrix with a 3x3 rotation matrix :math:`\boldsymbol R`
and a column vector :math:`\boldsymbol t` that represents the translation.
It is also sometimes called the homogeneous representation of a transformation.
All transformation matrices of this form generate the special Euclidean group
:math:`SE(3)`, that is,

.. math::

    SE(3) = \{ \boldsymbol{T} = \left(
    \begin{array}{cc}
    \boldsymbol{R} & \boldsymbol{t}\\
    \boldsymbol{0} & 1
    \end{array}
    \right) \in \mathbb{R}^{4 \times 4}
    | \boldsymbol{R} \in SO(3), \boldsymbol{t} \in \mathbb{R}^3 \}.

pytransform3d uses a numpy array of shape (4, 4) to represent transformation
matrices and typically we use the variable name A2B for a transformation
matrix, where A corrsponds to the frame from which it transforms and B to
the frame to which it transforms.

It is possible to transform position vectors or direction vectors with it.
Position vectors are represented as a column vector
:math:`\left( x,y,z,1 \right)^T`.
This will activate the translation part of the transformation in a matrix
multiplication (see :func:`~pytransform3d.transformations.vector_to_point`).
When we transform a direction vector, we want to deactivate the translation by
setting the last component to zero (see
:func:`~pytransform3d.transformations.vector_to_direction`):
:math:`\left( x,y,z,0 \right)^T`.

We can use a transformation matrix :math:`\boldsymbol T_{AB}` to transform a
point :math:`{_B}\boldsymbol{p}` from frame :math:`B` to frame :math:`A`:

.. math::

    \boldsymbol{T}_{AB}  {_B}\boldsymbol{p} =
    \left( \begin{array}{c}
        \boldsymbol{R} {_B}\boldsymbol{p} + \boldsymbol t\\
        1\\
    \end{array} \right).

You can use :func:`~pytransform3d.transformations.transform` to apply a
transformation matrix to a homogeneous vector.

-----------------------
Position and Quaternion
-----------------------

An alternative to transformation matrices is the representation in a
7-dimensional vector that consists of the translation and a rotation
quaternion:

.. math::

    \left( x, y, z, q_w, q_x, q_y, q_z \right)^T

This representation is more compact than a transformation matrix and is
particularly useful if you want to represent a sequence of poses in
a 2D array.

pytransform3d uses a numpy array of shape (7,) to represent position and
quaternion and typically we use the variable name pq.

----------------
Screw Parameters
----------------

.. plot:: ../../examples/plots/plot_screw.py

Just like any rotation can be expressed as a rotation by an angle about a
3D unit vector, any transformation (rotation and translation) can be expressed
by a motion along a screw axis [2]_ [3]_ [4]_. The **screw parameters** that
describe a screw axis include a point vector :math:`\boldsymbol{q}` through
which the screw axis passes, a (unit) direction vector
:math:`\hat{\boldsymbol{s}}` that
indicates the direction of the axis, and the pitch :math:`h`. The pitch
represents the ratio of translation and rotation. A screw motion translates
along the screw axis and rotates about it.

pytransform3d uses two vectors q and `s_axis` of shape (3,) and a scalar
h to represent the parameters of a screw.

.. image:: ../_static/screw_axis.png
   :alt: Screw axis
   :width: 50%
   :align: center

----------
Screw Axis
----------

A **screw axis** is typically represented by
:math:`\mathcal{S} = \left[\begin{array}{c}\boldsymbol{\omega}\\\boldsymbol{v}\end{array}\right] \in \mathbb{R}^6`,
where either

1. :math:`||\boldsymbol{\omega}|| = 1` or
2. :math:`||\boldsymbol{\omega}|| = 0` and :math:`||\boldsymbol{v}|| = 1`
   (only translation).

pytransform3d uses a numpy array of shape (6,) to represent a screw axis
and typically we use the variable name S or `screw_axis`.

In case 1, we can compute the screw axis from screw parameters
:math:`(\boldsymbol{q}, \hat{\boldsymbol{s}}, h)` as

.. math::

    \mathcal{S} = \left[ \begin{array}{c}\hat{\boldsymbol{s}} \\ \boldsymbol{q} \times \hat{\boldsymbol{s}} + h \hat{\boldsymbol{s}}\end{array} \right]

In case 2, :math:`h` is infinite and we directly translate along :math:`\hat{\boldsymbol{s}}`.

-----------------------
Exponential Coordinates
-----------------------

By multiplication with an additional parameter :math:`\theta` we can then
define a complete transformation through its exponential coordinates
:math:`\mathcal{S} \theta = \left[\begin{array}{c}\boldsymbol{\omega}\theta\\\boldsymbol{v}\theta\end{array}\right] \in \mathbb{R}^6`.
This is a minimal representation as it only needs 6 values.

pytransform3d uses a numpy array of shape (6,) to represent a exponential
coordinates of transformation and typically we use the variable name Stheta.

.. warning::

    Note that we use the screw theory definition of exponential coordinates
    and :math:`se(3)` (see next section) used by Paden (1985), Lynch and Park
    (2017), and Corke (2017). They separate the parameter :math:`\theta` from
    the screw axis. Additionally, they use the first three components to encode
    rotation and the last three components to encode translation. There is an
    alternative definition used by Eade (2017) and Sola et al. (2018). They use
    a different order of the 3D vector components and they do not separate
    :math:`\theta` from the screw axis in their notation.

---------------------------
Logarithm of Transformation
---------------------------

Alternatively, we can represent a screw axis :math:`\mathcal S` in a matrix

.. math::

    \left[\mathcal S\right]
    =
    \left( \begin{array}{cc}
        \left[\boldsymbol{\omega}\right] & \boldsymbol v\\
        \boldsymbol 0 & 0\\
    \end{array} \right)
    =
    \left(
    \begin{matrix}
    0 & -\omega_3 & \omega_2 & v_1\\
    \omega_3 & 0 & -\omega_1 & v_2\\
    -\omega_2 & \omega_1 & 0 & v_3\\
    0 & 0 & 0 & 0\\
    \end{matrix}
    \right)
    \in se(3) \subset \mathbb{R}^{4 \times 4}

that contains the cross-product matrix of its orientation part and its
translation part. This is the **matrix representation of a screw axis** and
we will also refer to it as **screw matrix** in the API.

pytransform3d uses a numpy array of shape (4, 4) to represent a screw matrix
and typically we use the variable name `screw_matrix`.

By multiplication with :math:`\theta` we can again generate a full
description of a transformation
:math:`\left[\mathcal{S}\right] \theta \in se(3)`, which is the **matrix
logarithm of a transformation matrix** and :math:`se(3)` is the Lie
algebra of Lie group :math:`SE(3)`.

pytransform3d uses a numpy array of shape (4, 4) to represent the logarithm
of a transformation and typically we use the variable name `transform_log`.

-----
Twist
-----

We call spatial velocity (translation and rotation) **twist**. Similarly
to the matrix logarithm, a twist :math:`\mathcal{V} = \mathcal{S} \dot{\theta}`
is described by a screw axis :math:`S` and a scalar :math:`\dot{\theta}`
and :math:`\left[\mathcal{V}\right] = \left[\mathcal{S}\right] \dot{\theta} \in se(3)`
is the matrix representation of a twist.

----------------
Dual Quaternions
----------------

Similarly to unit quaternions for rotations, unit dual quaternions are
an alternative to represent transformations [5]_ [6]_ [7]_. They support
similar operations as transformation matrices.

A dual quaternion consists of a real quaternion and a dual quaternion:

.. math::

    \boldsymbol{p} + \epsilon \boldsymbol{q} = p_w + p_x i + p_y j + p_z k + \epsilon (q_w + q_x i + q_y j + q_z k),

where :math:`\epsilon^2 = 0` and :math:`\epsilon \neq 0`.
We use unit dual quaternions to represent
transformations. In this case, the real quaternion is a unit quaternion
and the dual quaternion is orthogonal to the real quaternion.
The real quaternion is used to represent the rotation and the dual
quaternion contains information about the rotation and translation.

Dual quaternions support similar operations as transformation matrices
(inversion through the conjugate of the two individual quaternions
:func:`~pytransform3d.transformations.dq_q_conj`, concatenation
through :func:`~pytransform3d.transformations.concatenate_dual_quaternions`,
and transformation of a point by
:func:`~pytransform3d.transformations.dq_prod_vector`),
they can be renormalized efficiently (with
:func:`~pytransform3d.transformations.check_dual_quaternion`, and
interpolation between two dual quaternions is possible (with
:func:`~pytransform3d.transformations.dual_quaternion_sclerp`).

.. warning::

    The unit dual quaternions :math:`\boldsymbol{p} + \epsilon \boldsymbol{q}`
    and :math:`-\boldsymbol{p} - \epsilon \boldsymbol{q}` represent exactly
    the same transformation.

The reason for this ambiguity is that the real quaternion
:math:`\boldsymbol{p}` represents the orientation component, the dual
quaternion encodes the translation component as
:math:`\boldsymbol{q} = 0.5 \boldsymbol{t} \boldsymbol{p}`, where
:math:`\boldsymbol{t}` is a quaternion with the translation in the vector
component and the scalar 0, and rotation quaternions have the same ambiguity.

----------
References
----------

.. [1] Lynch, K. M., Park, F. C. (2017). Modern Robotics.
   http://hades.mech.northwestern.edu/index.php/Modern_Robotics
.. [2] Corke, P. (2017). Robotics, Vision and Control, 2nd Edition,
   https://link.springer.com/book/10.1007/978-3-319-54413-7
.. [3] Eade, E. (2017). Lie Groups for 2D and 3D Transformations.
   https://ethaneade.com/lie.pdf
.. [4] Sola, J., Deray, J., Atchuthan, D. (2018).
   A micro Lie theory for state estimation in robotics. Technical Report.
   http://www.iri.upc.edu/files/scidoc/2089-A-micro-Lie-theory-for-state-estimation-in-robotics.pdf
.. [5] Wikipedia: Dual Quaternion.
   https://en.wikipedia.org/wiki/Dual_quaternion
.. [6] Jia, Y.-B.: Dual Quaternions.
   https://faculty.sites.iastate.edu/jia/files/inline-files/dual-quaternion.pdf
.. [7] Kenwright, B. A Beginners Guide to Dual-Quaternions.
   http://wscg.zcu.cz/WSCG2012/!_WSCG2012-Communications-1.pdf
