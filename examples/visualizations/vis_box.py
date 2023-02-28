"""
========
Plot Box
========
"""
import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle


rng = np.random.default_rng(42)
fig = pv.figure()
A2B = transform_from(
    R=matrix_from_axis_angle(random_axis_angle(rng)),
    p=np.zeros(3))
fig.plot_box(size=[0.2, 0.5, 1], A2B=A2B)
fig.plot_transform(A2B=A2B)
fig.view_init()
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
