========================
Modeling Transformations
========================

With many transformations it is not easy to keep track of the right sequence.
Here are some simple tricks that you can use to keep track of transformations
and make to always concatenate them in the correct way.

.. image:: ../_static/transformation_modeling.png
   :alt: Three frames
   :align: center

|

When modeling transformations in mathematical equations we often sequence them
from right to left (extrinsic convention).
Here it makes most sense to give names like :math:`\boldsymbol T_{BA}` for
transformations that maps points **from frame** :math:`A` **to frame**
:math:`B`, so we can easily recognize that

.. math::

    \boldsymbol T_{CA} = \boldsymbol T_{CB} \boldsymbol T_{BA}

when we read the names of the frames from right to left.
Now we can use the transformation matrix :math:`\boldsymbol T_{CA}` to transform
a point :math:`_A\boldsymbol{p}` from frame :math:`A` to frame
:math:`C` by multiplication :math:`_C\boldsymbol{p} = \boldsymbol{T}_{CA}\,_A\boldsymbol{p}`.

In code this might look differently. Here we should prefer the notation `A2B`
for a transformation from frame `A` to frame `B`.

.. code-block::

    from pytransform3d.transformations import concat
    A2B = ...  # transformation from frame A to frame B
    B2C = ...  # transformation from frame B to frame C
    A2C = concat(A2B, B2C)

Now we can verify that the new transformation matrix correctly transforms from
frame `A` to frame `C` if we read from left to right. The function `concat`
will correctly apply the transformation `B2C` after `A2B`. If we want to transform
a point from `C` to `A` we can now use

.. code-block::

    from pytransform3d.transformations import vector_to_point, transform
    p_in_A = vector_to_point(...)  # point in frame A
    p_in_C = transform(A2C, p_in_A)

For more complex cases the :class:`~pytransform3d.transform_manager.TransformManager`
can help.
