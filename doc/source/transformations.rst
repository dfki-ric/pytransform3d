=========================
3D Transformations: SE(3)
=========================

The group of all transformations in the 3D Cartesian space is :math:`SE(3)`.
Transformations consist of a rotation and a translation. Those can be
represented in multiple different ways. One of the most convenient ways
to represent transformations are transformation matrices.
A transformation matrix is a 4x4 matrix of the form

.. math::

    \boldsymbol M =
    \left( \begin{array}{cc}
        \boldsymbol R & \boldsymbol t\\
        \boldsymbol 0 & 1\\
    \end{array} \right)

It is a partitioned matrix with a 3x3 rotation matrix :math:`\boldsymbol R`
and a column vector :math:`\boldsymbol t` that represents the translation.
It is also sometimes called the homogeneous representation of a transformation.

It is possible to transform position vectors or direction vectors with it.
Position vectors are represented as a column vector
:math:`\left( x,y,z,1 \right)^T`.
This will activate the translation part of the transformation in a matrix
multiplication. When we transform a direction vector, we want to deactivate
the translation by setting the last component to zero:
:math:`\left( x,y,z,0 \right)^T`. For example, transforming a position
vector :math:`p` will give the following result:

.. math::

    \boldsymbol M \boldsymbol p =
    \left( \begin{array}{cc}
        \boldsymbol R \boldsymbol p + \boldsymbol t\\
        1\\
    \end{array} \right)