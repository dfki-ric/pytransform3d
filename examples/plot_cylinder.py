""""
==========================
Plot Transformed Cylinders
==========================

Plots surfaces of transformed cylindrical shells.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import transform_from, plot_transform
from pytransform3d.rotations import random_axis_angle, matrix_from_axis_angle
from pytransform3d.plot_utils import plot_cylinder


random_state = np.random.RandomState(42)
A2B = transform_from(
    R=matrix_from_axis_angle(random_axis_angle(random_state)),
    p=random_state.randn(3))
ax = plot_cylinder(length=1.0, radius=0.3, thickness=0.1, ax_s=1.5,
                   wireframe=False, alpha=0.2)
plot_transform(ax=ax, A2B=np.eye(4), s=0.3, lw=3)
plot_cylinder(ax=ax, length=1.0, radius=0.3, thickness=0.1, A2B=A2B,
              wireframe=False, alpha=0.2)
plot_transform(ax=ax, A2B=A2B, s=0.3, lw=3)
plt.show()