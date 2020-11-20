"""
================
Animate Rotation
================

Animates a rotation about the x-axis.
"""
print(__doc__)


import pytransform3d.visualizer as pv
from pytransform3d.rotations import *


def animation_callback(step, n_frames, frame):
    angle = 2.0 * np.pi * (step + 1) / n_frames
    R = matrix_from_angle(0, angle)
    A2B = np.eye(4)
    A2B[:3, :3] = R
    frame.set_data(A2B)
    return frame


fig = pv.figure()
frame = fig.plot_basis(R=np.eye(3), s=0.5)
fig.view_init()

n_frames = 100
fig.animate(animation_callback, n_frames, fargs=(n_frames, frame), loop=True)
fig.show()