.. pytransform documentation master file, created by
   sphinx-quickstart on Thu Nov 20 21:01:30 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========
pytransform
===========

Contents:

.. toctree::
   :maxdepth: 2

   api

========
Notation
========

We will use the notation

.. math::

    _A\boldsymbol{t}_{BC}

to represent a vector from frame B to frame C expressed in frame A.

**References**

* http://paulfurgale.info/news/2014/6/9/representing-robot-pose-the-good-the-bad-and-the-ugly


===========
Orientation
===========

The group of all rotations in the 3D Cartesian space is called :math:`SO(3)`.
The minimum number of components that are required to describe any rotation
from :math:`SO(3)` is 3. However, there is no representation that is
non-redundant, continuous and free of singularities. We will now take a closer
look at competing representations of rotations and the orientations they can
describe.

.. plot:: ../../examples/plot_compare_rotations.py
    :include-source:

---------------
Rotation Matrix
---------------

The most practical representation of orientation is a rotation matrix

.. math::

    \boldsymbol R =
    \left( \begin{array}{ccc}
        r_{11} & r_{12} & r_{13}\\
        r_{21} & r_{22} & r_{23}\\
        r_{31} & r_{32} & r_{33}\\
    \end{array} \right)

Note that

* this is a non-minimal representation for orientations because we have 9
  values but only 3 degrees of freedom
* :math:`\boldsymbol R` must be orthonormal
* :math:`\boldsymbol R^T = \boldsymbol R^{-1}`
* :math:`det(\boldsymbol R) = 1`

We can use a rotation matrix :math:`\boldsymbol R_{AB}` to transform a point
:math:`_B\boldsymbol{p} := _B\boldsymbol{t}_{BP}` from frame :math:`B` to frame
:math:`A`.

.. warning::

    There are two different conventions on how to use rotation matrices to
    apply a rotation. They can either be multiplied from the left side to
    a vector or from the right side. Here, **we will assume that rotation
    matrices are multiplied from the left side**.

This means that we rotate a point :math:`_B\boldsymbol{p}` by

.. math::

    _A\boldsymbol{p} = \boldsymbol{R}_{ABB} \boldsymbol{p}

This is called **linear map**.

We can see that each column of such a rotation matrix is a basis vector
of frame :math:`A` with respect to frame :math:`B`.

We can plot the basis vectors of an orientation to visualize it.

.. note::

    When plotting basis vectors it is a convention to use red for the x-axis,
    green for the y-axis and blue for the z-axis (RGB for xyz).

Here, we can see orientation represented by the rotation matrix

.. math::

    \boldsymbol R =
    \left( \begin{array}{ccc}
        1 & 0 & 0\\
        0 & 1 & 0\\
        0 & 0 & 1\\
    \end{array} \right)

.. plot::
    :include-source:

    from pytransform.rotations import plot_basis
    plot_basis()

We can easily chain multiple rotations: we can apply the rotation defined
by :math:`\boldsymbol R_{AB}` after the rotation :math:`\boldsymbol R_{BC}`
by applying the rotation

.. math::

    \boldsymbol R_{AC} = \boldsymbol R_{AB} \boldsymbol R_{BC}.

**Pros**

* It is easy to apply rotations on point vectors
* Concatenation of rotations is trivial
* You can directly read the basis vectors from the columns

**Cons**

* We use 9 values for 3 degrees of freedom

----------
Axis-Angle
----------

Each rotation can be represented by a single rotation around one axis.

.. plot:: ../../examples/plot_axis_angle.py
    :include-source:

The axis can be represented as a three-dimensional unit vector and the angle
by a scalar:

.. math::

    \left( \boldsymbol{\hat{e}}, \theta \right) = \left( \left( \begin{array}{c}e_x\\e_y\\e_z\end{array} \right), \theta \right)

It is possible to write this in a more compact way as a rotation vector:

.. math::

    \boldsymbol{v} = \theta \boldsymbol{\hat{e}}

**Pros**

* Minimal representation (as rotation vector)
* It is easy to interpret the representation (as axis and angle)

**Cons**

* Concatenation involves conversion to another representation

------------
Euler Angles
------------

A complete rotation can be split into three rotations around basis vectors.

.. warning::

    There are 24 different conventions for defining euler angles. We will
    only use the XYZ convention and the ZYX convention.

.. plot:: ../../examples/plot_euler_angles.py
    :include-source:

**Pros**

* Minimal representation

**Cons**

* 24 different conventions
* Singularities (gimbal lock)

-----------
Quaternions
-----------

The unit quaternion space :math:`S^3` can be used to represent orientations.
To do that, we use an encoding based on the rotation axis and angle.

A rotation quaternion is a four-dimensional unit vector (versor)

.. math::

    \boldsymbol{\hat{q}} =
    \left( \begin{array}{c}
        \cos \frac{\theta}{2}\\
        e_x \sin \frac{\theta}{2}\\
        e_y \sin \frac{\theta}{2}\\
        e_z \sin \frac{\theta}{2}\\
    \end{array} \right)

.. warning::

    The scalar component of a quaternion is sometimes the first element and
    sometimes the last element of the versor. We will always use the first
    element to store the scalar component.

**Pros**

* More compact than the matrix representation and less susceptible to
  round-off errors
* The quaternion elements vary continuously over the unit sphere in
  :math:`\mathbb{R}^4` as the orientation changes, avoiding discontinuous
  jumps (inherent to three-dimensional parameterizations)
* Expression of the rotation matrix in terms of quaternion parameters
  involves no trigonometric functions
* Concatenation is simple with the quaternion product

**Cons**

* The representation is not straightforward to interpret
