"""
============
Animate Line
============

Animates a line.
"""
print(__doc__)


import pytransform3d.visualizer as pv
from pytransform3d.rotations import *


def animation_callback(step, n_frames, line):
    t = step / n_frames
    P = np.empty((100, 3))
    for d in range(P.shape[1]):
        P[:, d] = np.linspace(0, t, len(P))
    line.set_data(P)
    return line


fig = pv.figure()
P = np.zeros((100, 3))
colors = np.empty((100, 3))
for d in range(colors.shape[1]):
    colors[:, d] = np.linspace(0, 255, len(colors))
line = fig.plot(P, colors)
frame = fig.plot_basis(R=np.eye(3), s=0.5)
fig.view_init()

n_frames = 100
fig.animate(animation_callback, n_frames, fargs=(n_frames, line), loop=True)
fig.show()