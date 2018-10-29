"""
=================
Camera Projection
=================

We can see the camera coordinate frame and a grid of points in the camera
coordinate system which will be projected on the sensor. From the coordinates
on the sensor we can compute the corresponding pixels.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.camera import make_world_grid, cam2sensor, sensor2img


focal_length = 0.2
sensor_size = (0.2, 0.15)
image_size = (640, 480)

plt.figure(figsize=(12, 5))
ax = plt.subplot(121, projection="3d", aspect="equal")
ax.set_title("Grid in 3D camera coordinate system")
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.set_zlim((0, 2))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

cam_grid = make_world_grid(n_points_per_line=11) - np.array([0, 0, -2, 0])
img_grid = cam_grid * focal_length

c = np.arange(len(cam_grid))
ax.scatter(cam_grid[:, 0], cam_grid[:, 1], cam_grid[:, 2], c=c)
ax.scatter(img_grid[:, 0], img_grid[:, 1], img_grid[:, 2], c=c)
plot_transform(ax)

sensor_grid = cam2sensor(cam_grid, focal_length)
img_grid = sensor2img(sensor_grid, sensor_size, image_size)
ax = plt.subplot(122, aspect="equal")
ax.set_title("Grid in 2D image coordinate system")
ax.scatter(img_grid[:, 0], img_grid[:, 1], c=c)
ax.set_xlim((0, image_size[0]))
ax.set_ylim((0, image_size[1]))

plt.show()
