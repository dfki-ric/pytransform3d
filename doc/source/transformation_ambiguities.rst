==========================
Transformation Ambiguities
==========================

There are lots of ambiguities in the world of transformations. We try to
explain them all here.

---------------------------------
Active vs. Passive Transformation
---------------------------------

An active transformation

* changes the physical position of an object
* can be defined in the absence of a coordinate system or does not change the
  current coordinate system
* is used exclusively by mathematicians

Another name for active transformation is alibi transformation.

A passive transformation

* changes the coordinate system in which the object is described
* does not change the object
* could be used by physicists and engineers

Another name for passive transformation is alias transformation.

-------------------------------------
Source in Target vs. Source to Target
-------------------------------------

A transformations is defined between two frames: source and target, origin
and target, or parent and child. You can also say it transforms from some
frame to another frame.
However, it is not always clear how to interpret the source and the target.

The first option would be to assume that the source is the base frame in
which we represent the target frame and the transformation gives us the
translation and rotation to get the location of the target in the source.

The second option assumes that the transformation transforms data, for example,
points from the source frame to the target frame.

Whenever you hear that there is a transformation from some frame to another
frame, make sure you understand what is meant.

--------------------------------
Right-multiply vs. Left-multiply
--------------------------------



----------------------------------------------
Right-handed vs. Left-handed Coordinate System
----------------------------------------------

We typically use a right-handed coordinate system, that is, the x-, y- and
z-axis are aligned in a specific way. The name comes from the way how the
fingers are attached to the human hand. Try to align your thumb with the
imaginary x-axis, your index finger with the y-axis, and your middle finger
with the z-axis. It is possible to do this with a right hand in a
right-handed system and with the left hand in a left-handed system.

.. raw:: html

    <table>
    <tr><td>Right-handed</td><td>Left-handed</td></tr>
    <tr>
    <td>

.. plot:: ../../examples/plot_convention_right_hand_coordinate_system.py
    :width: 400px

.. raw:: html

    </td>
    <td>

.. plot:: ../../examples/plot_convention_left_hand_coordinate_system.py
    :width: 400px

.. raw:: html

    </td>
    </tr>
    <table>