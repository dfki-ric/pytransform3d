=================
Transform Manager
=================

It is sometimes very difficult to have an overview of all the transformations
that are required to obtain one specific transformation. Suppose you have
a robot with a camera that can observe the robot's end-effector and an object
that we want to manipulate. We would like to know the position of the
end-effector in the object's frame so that we can control it. The
:class:`~pytransform3d.transform_manager.TransformManager` can handle this
for you.

.. plot:: ../../examples/plot_transform_manager.py
    :include-source:

We can also export the underlying graph structure as a PNG with::

    tm.write_png(filename)

.. image:: _static/graph.png
    :width: 50%
    :align: center


A subclass of the transformation manager is the
:class:`~pytransform3d.urdf.UrdfTransformManager` which can be used to load
robot definitions from URDF files. An example with a simple robot can be seen
in the following example.

.. plot:: ../../examples/plot_urdf.py
    :include-source:

The same class can be used to display primitive collision objects (no meshes)
or visuals from URDF files:

.. plot:: ../../examples/plot_collision_objects.py
    :include-source:
