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

.. image:: ../_static/camera.png
   :alt: Camera
   :align: center
   :width: 60%

|

Note that light passes through a pinhole in a real pinhole camera before it
will be measured from the sensor so that pixels will be mirrored in the x-y
plane. The sensor that we show here actually corresponds to the virtual
image plane.

The Example :ref:`sphx_glr__auto_examples_plots_plot_camera_with_image.py`
shows how a grid is projected on an image with the function
:func:`~pytransform3d.camera.world2image`.

.. figure:: ../_auto_examples/plots/images/sphx_glr_plot_camera_with_image_001.png
   :target: ../_auto_examples/plots/plot_camera_with_image.html
   :align: center

Extrinsic and intrinsic camera parameters can be visualized in the following
way. The extrinsic camera parameters are fully determined by a transform
from world coordinates to camera coordinates or by the pose of the camera in
the world. In this illustration, the point indicates the camera center /
center of projection, which is the position component of the pose. The
orientation determines the direction to and orientation of the virtual image
plane. The arrow at the top of the virtual image plane shows the up direction
of the image.

The field of view is determined from the intrinsic camera parameters. These
are given by a matrix

.. math::

    \left( \begin{array}{ccc}
    f_x & 0 & c_x\\
    0 & f_y & c_y\\
    0 & 0 & 1
    \end{array} \right),

where :math:`f_x, f_y` are focal lengths and :math:`c_x, c_y` is the position
of the camera center. Together with the image size we can determine the field
of view. Values of the intrinsic camera matrix and the image size can be given
in pixels or meters to generate the following visualization with
:func:`~pytransform3d.camera.plot_camera` (see Example
:ref:`sphx_glr__auto_examples_plots_plot_camera_3d.py`).

.. figure:: ../_auto_examples/plots/images/sphx_glr_plot_camera_3d_001.png
   :target: ../_auto_examples/plots/plot_camera_3d.html
   :align: center

You can use this to display a trajectory of camera poses (see Example
:ref:`sphx_glr__auto_examples_plots_plot_camera_trajectory.py`).

.. figure:: ../_auto_examples/plots/images/sphx_glr_plot_camera_trajectory_001.png
   :target: ../_auto_examples/plots/plot_camera_trajectory.html
   :align: center
