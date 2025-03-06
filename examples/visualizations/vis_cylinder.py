"""
===============================
Visualize Transformed Cylinders
===============================

Plots transformed cylinders.
"""

import numpy as np

import pytransform3d.visualizer as pv
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle
from pytransform3d.transformations import transform_from

fig = pv.figure()
rng = np.random.default_rng(42)
A2B = transform_from(
    R=matrix_from_axis_angle(random_axis_angle(rng)),
    p=rng.standard_normal(size=3),
)
fig.plot_cylinder(length=1.0, radius=0.3)
fig.plot_transform(A2B=np.eye(4))
fig.plot_cylinder(length=1.0, radius=0.3, A2B=A2B)
fig.plot_transform(A2B=A2B)
fig.view_init()
fig.set_zoom(2)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
