========================
Managing Transformations over Time
========================

In applications, where the transformations between coordinate frames are 
changing over time, prior to using 
:class:`~pytransform3d.transform_manager.TransformManager` 
you should consider interpolating between samples.

We can visualize the problem in the figure below. The tranformation graph with 
3 coordinate systems: W (world), A, B is visualized over time. Each circle 
represents a sample (measurement) holding the transformation from the parent 
to the child frame.

Let's assume we want to inspect the situation at the timestep :math:`t_q` (q=query). 
Further we want to transform points from A to B. As shown in the previous example
:class:`~pytransform3d.transform_manager.TransformManager` is helpful to deal 
with transformation graphs.

However, prior to creating the manager, we have to interpolate the transformation streams
at a timestep of interest. This is visualized with small circles filled with color of
the according transformation direction.

.. figure:: _static/tf-trafo-over-time.png
    :width: 60%
    :align: center

In this example, the screw linear interpolation (ScLERP) will be used
(which operates on dual quaternions, refer to 
:func:`~pytransform3d.transformations.pq_from_dual_quaternion`). A dual 
quaternion representation holds both the translation and rotation information.

.. literalinclude:: ../../examples/plots/plot_interpolation_for_transform_manager.py
   :language: python
