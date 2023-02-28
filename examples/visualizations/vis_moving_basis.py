"""
================
Animate Rotation
================

Animates a rotation about the x-axis.
"""
import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d import rotations as pr


def animation_callback(step, n_frames, frame):
    angle = 2.0 * np.pi * (step + 1) / n_frames
    R = pr.passive_matrix_from_angle(0, angle)
    A2B = np.eye(4)
    A2B[:3, :3] = R
    frame.set_data(A2B)
    return frame


fig = pv.figure(width=500, height=500)
frame = fig.plot_basis(R=np.eye(3), s=0.5)
fig.view_init()

n_frames = 100
if "__file__" in globals():
    fig.animate(
        animation_callback, n_frames, fargs=(n_frames, frame), loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
