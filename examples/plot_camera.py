import numpy as np
import matplotlib.pyplot as plt
from pytransform.rotations import *
from pytransform.transformations import *
from pytransform.camera import *


n_points = 50

world_grid_x = np.vstack(
    [np.array([np.linspace(-1, 1, n_points),
               np.linspace(y, y, n_points),
               np.zeros(n_points),
               np.ones(n_points)]).T
     for y in np.linspace(-1, 1, 11)])
world_grid_y = np.vstack(
    [np.array([np.linspace(x, x, n_points),
               np.linspace(-1, 1, n_points),
               np.zeros(n_points),
               np.ones(n_points)]).T
     for x in np.linspace(-1, 1, 11)])
world_grid = np.vstack((world_grid_x, world_grid_y))


cam2world = rotate_transform(np.eye(4), matrix_from_euler_xyz([np.pi / 8, 0, 0]))
cam2world = translate_transform(cam2world, [0, -1, 0.5])
size_image = np.array([640, 480])
center_image = size_image / 2
focal_length = 100.0
kappa = 0.0

image_grid = world2image(cam2world, world_grid, center_image, focal_length,
                         kappa)

plt.figure()
ax = plt.subplot(111, projection="3d", aspect="equal")
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

plt.figure()
plt.xlim(0, size_image[0])
plt.ylim(0, size_image[1])
plt.scatter(size_image[0] - image_grid[:, 0],
            size_image[1] - image_grid[:, 1])
plt.scatter(size_image[0] - image_grid[-1, 0],
            size_image[1] - image_grid[-1, 1], color="r")

plt.show()
