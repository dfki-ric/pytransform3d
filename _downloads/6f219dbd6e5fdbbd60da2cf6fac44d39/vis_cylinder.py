"""
===============================
Visualize Transformed Cylinders
===============================

Plots transformed cylinders.
"""
print(__doc__)


import numpy as np
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle
import pytransform3d.visualizer as pv


fig = pv.figure()
random_state = np.random.RandomState(42)
A2B = transform_from(
    R=matrix_from_axis_angle(random_axis_angle(random_state)),
    p=random_state.randn(3))
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
