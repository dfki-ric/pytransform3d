=================
Transform Manager
=================

It is sometimes very difficult to have an overview of all the transformations
that are required to obtain one specific transformation. Suppose you have
a robot with a camera that can observe the robot's end-effector and an object
that we want to manipulate. We would like to know the position of the
end-effector in the object's frame so that we can control it.

.. plot:: ../../examples/plot_transform_manager.py
    :include-source: