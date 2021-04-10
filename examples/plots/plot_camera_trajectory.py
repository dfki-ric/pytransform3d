"""TODO description of example"""
print(__doc__)


import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.camera as pc
from cycler import cycle


BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "pytransform3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

with open(os.path.join(
        data_dir, "reconstruction_camera_matrix.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    intrinsic_matrix = np.array([[float(entry) for entry in row]
                                 for row in reader])

with open(os.path.join(data_dir, "reconstruction_odometry.csv"), "r") as f:
    reader = csv.reader(f, delimiter=",")
    next(reader)  # ignore column labels
    trajectory = np.array([[float(entry) for entry in row] for row in reader])

q_scalar = trajectory[:, -1]
q_vec = trajectory[:, 3:6]
trajectory[:, 3] = q_scalar
trajectory[:, 4:] = q_vec

# TODO check frame orientation
H = ptr.transforms_from_pqs(trajectory)

ax = pt.plot_transform(s=0.3)
ax = ptr.plot_trajectory(ax, P=trajectory, s=0.1)

image_size = np.array([1920, 1440])  # in pixels
sensor_size = np.array([0.036, 0.024])  # in meters
factor = sensor_size / image_size
intrinsic_matrix[:2, :2] *= np.eye(2).dot(factor)
intrinsic_matrix[:2, 2] *= factor

key_frames_indices = np.linspace(0, len(trajectory) - 1, 10, dtype=int)
colors = cycle("rgb")
for i, c in zip(key_frames_indices, colors):
    pc.plot_camera(ax, intrinsic_matrix, H[i], sensor_size=sensor_size,
                   virtual_image_distance=0.2, c=c)

pos_min = np.min(trajectory[:, :3], axis=0)
pos_max = np.max(trajectory[:, :3], axis=0)
center = (pos_max + pos_min) / 2.0
max_half_extent = max(pos_max - pos_min) / 2.0
ax.set_xlim((center[0] - max_half_extent, center[0] + max_half_extent))
ax.set_ylim((center[1] - max_half_extent, center[1] + max_half_extent))
ax.set_zlim((center[2] - max_half_extent, center[2] + max_half_extent))
plt.show()
