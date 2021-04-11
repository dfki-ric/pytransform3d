======
Camera
======

When we know the 3D position of a point in the world we can easily compute
where we would see it in a camera image with the pinhole camera model.
However, we have to know some parameters of the camera:

* camera pose
* focal length :math:`f`
* sensor width and height
* image width and height

.. image:: _static/camera.png
   :alt: Camera
   :align: center
   :width: 60%

|

Note that light passes through a pinhole in a real pinhole camera before it
will be measured from the sensor so that pixels will be mirrored in the x-y
plane. The sensor that we show here actually corresponds to the virtual
image plane.

The following example shows how a grid is projected on an image.

.. plot:: ../../examples/plots/plot_camera_with_image.py

|

Camera poses and configurations can be visualized in three dimensions
in the following way. The arrow at the top shows the up direction of
the image. The z-axis points from the camera center to a virtual image
plane. The field of view is determined from the intrinsic camera
matrix.

.. plot:: ../../examples/plots/plot_camera_3d.py

You can use this to display a trajectory of camera poses.

.. plot:: ../../examples/plots/plot_camera_trajectory.py
