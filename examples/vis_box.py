"""
========
Plot Box
========
"""
import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.transformations import transform_from
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle


random_state = np.random.RandomState(42)
fig = pv.figure()
A2B = transform_from(
    R=matrix_from_axis_angle(random_axis_angle(random_state)),
    p=np.zeros(3))
fig.plot_box(size=[0.2, 0.5, 1], A2B=A2B)
fig.plot_transform(A2B=A2B)
fig.view_init()
fig.show()
