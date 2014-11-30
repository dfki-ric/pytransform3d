import numpy as np
import matplotlib.pyplot as plt
from pytransform.rotations import *
from pytransform.transformations import *
from pytransform.camera import *


cam2world = transform_from(matrix_from_euler_xyz([0.35 * np.pi, 0, 0]),
                           [0, -1, 0.5])
focal_length = 0.0036
sensor_size = (0.00367, 0.00274)
image_size = (640, 480)

world_grid = make_world_grid()
image_grid = world2image(world_grid, cam2world, sensor_size, image_size,
                         focal_length)

plt.figure(figsize=(12, 5))
ax = plt.subplot(121, projection="3d", aspect="equal")
ax.set_xlim((-1, 1))
ax.set_ylim((-1, 1))
ax.set_zlim((-1, 1))
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plot_transform(ax)
plot_transform(ax, A2B=cam2world)
ax.scatter(world_grid[:, 0], world_grid[:, 1], world_grid[:, 2])
ax.scatter(world_grid[-1, 0], world_grid[-1, 1], world_grid[-1, 2], color="r")
for p in world_grid[::10]:
    ax.plot([p[0], cam2world[0, 3]],
            [p[1], cam2world[1, 3]],
            [p[2], cam2world[2, 3]], alpha=0.2)

ax = plt.subplot(122, aspect="equal")
plt.xlim(0, image_size[0])
plt.ylim(0, image_size[1])
plt.scatter(image_grid[:, 0], image_grid[:, 1])
plt.scatter(image_grid[-1, 0], image_grid[-1, 1], color="r")

plt.show()
