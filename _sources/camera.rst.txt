======
Camera
======

When we know the 3D position of a point in the world we can easily compute
where we would see it in a camera image. However, we have to know some
parameters of the camera:

* camera pose
* focal length :math:`f`
* sensor width and height
* image width and height

.. image:: _static/camera.svg
   :alt: Camera
   :align: center

The following example shows how a grid is projected on an image.

.. plot:: ../../examples/plot_camera.py
    :include-source: