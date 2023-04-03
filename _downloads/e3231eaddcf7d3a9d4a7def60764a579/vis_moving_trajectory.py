"""
==================
Animate Trajectory
==================

Animates a trajectory.
"""
import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.rotations import passive_matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat


def update_trajectory(step, n_frames, trajectory):
    progress = 1 - float(step + 1) / float(n_frames)
    H = np.zeros((100, 4, 4))
    H0 = transform_from(R_id, np.zeros(3))
    H_mod = np.eye(4)
    for i, t in enumerate(np.linspace(0, progress, len(H))):
        H0[:3, 3] = np.array([t, 0, t])
        H_mod[:3, :3] = passive_matrix_from_angle(2, 8 * np.pi * t)
        H[i] = concat(H0, H_mod)

    trajectory.set_data(H)
    return trajectory


n_frames = 200

fig = pv.figure()

H = np.empty((100, 4, 4))
H[:] = np.eye(4)
# set initial trajectory to extend view box
H[:, 0, 3] = np.linspace(-2, 2, len(H))
H[:, 1, 3] = np.linspace(-2, 2, len(H))
H[:, 2, 3] = np.linspace(0, 4, len(H))
trajectory = pv.Trajectory(H, s=0.2, c=[0, 0, 0])
trajectory.add_artist(fig)
fig.view_init()
fig.set_zoom(0.5)

if "__file__" in globals():
    fig.animate(
        update_trajectory, n_frames, fargs=(n_frames, trajectory), loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
