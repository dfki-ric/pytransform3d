========================================
Graphs of Time-Dependent Transformations
========================================

In applications, where the transformations between coordinate frames are
dynamic (i.e. changing over time), consider using
:class:`~pytransform3d.transform_manager.TemporalTransformManager`. In contrast
to the :class:`~pytransform3d.transform_manager.TransformManager`,
which deals with static transfomations, it provides an interface for the logic
needed to interpolate between transformation samples available over time.

We can visualize the lifetime of two dynamic transformations (i.e. 3 coordinate
systems) in the figure below. Each circle represents a sample (measurement)
holding the transformation from the parent to the child frame.

.. figure:: ../_static/tf-trafo-over-time.png
    :width: 60%
    :align: center

A common use-case is to transform points originating from system A to system B
at a specific point in time (i.e. :math:`t_q`, where :math:`q` refers to query)
Imagine two moving robots A & B reporting their observations between each
other.

--------------------------------------
Preparing the transformation sequences
--------------------------------------

First, you need to prepare the transfomation sequences using the 
:class:`~pytransform3d.transform_manager.NumpyTimeseriesTransform` class:

.. literalinclude:: ../../../examples/plots/plot_interpolation_for_transform_manager.py
   :language: python
   :lines: 43-56

In this example, the screw linear interpolation (ScLERP) will be used
(which operates on dual quaternions, refer to
:func:`~pytransform3d.transformations.pq_from_dual_quaternion`).

For more control, you may want to add your own implementation of the
abstract class :class:`~pytransform3d.transform_manager.TimeVaryingTransform`.

Next, you need to pass the transformations to an instance of
:class:`~pytransform3d.transform_manager.TemporalTransformManager`:

.. literalinclude:: ../../../examples/plots/plot_interpolation_for_transform_manager.py
   :language: python
   :lines: 58-61

------------------------------------
Transform between coordinate systems
------------------------------------

Finally, you can transform between coordinate systems at a particular time
:math:`t_q`:

.. literalinclude:: ../../../examples/plots/plot_interpolation_for_transform_manager.py
   :language: python
   :lines: 63-68

The coordinates of A's origin (blue diamond) transformed to B are visualized
in the plot below:

.. plot:: ../../examples/plots/plot_interpolation_for_transform_manager.py
