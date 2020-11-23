"""
==================
Animate Trajectory
==================

Animates a trajectory.
"""
print(__doc__)


import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.rotations import matrix_from_angle, R_id
from pytransform3d.transformations import transform_from, concat


def update_trajectory(step, n_frames, trajectory):
    progress = float(step + 1) / float(n_frames)
    H = np.zeros((100, 4, 4))
    H0 = transform_from(R_id, np.zeros(3))
    H_mod = np.eye(4)
    for i, t in enumerate(np.linspace(0, progress, len(H))):
        H0[:3, 3] = np.array([t, 0, t])
        H_mod[:3, :3] = matrix_from_angle(2, 8 * np.pi * t)
        H[i] = concat(H0, H_mod)

    trajectory.set_data(H)
    return trajectory


if __name__ == "__main__":
    n_frames = 200

    fig = pv.figure()

    H = np.zeros((100, 4, 4))
    H[:] = np.eye(4)
    trajectory = pv.Trajectory(H, s=0.2, c=[0, 0, 0])
    trajectory.add_trajectory(fig)
    fig.view_init()
    fig.set_zoom(8)

    fig.animate(update_trajectory, n_frames, fargs=(n_frames, trajectory), loop=True)
    fig.show()
