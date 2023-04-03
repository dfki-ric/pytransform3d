"""
=====================================
Axis-Angle Representation of Rotation
=====================================

Any rotation can be represented with a single rotation about some axis.
Here we see a frame that is rotated in multiple steps around a rotation
axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import (random_axis_angle, matrix_from_axis_angle,
                                     plot_basis, plot_axis_angle)


original = random_axis_angle(np.random.RandomState(5))
ax = plot_axis_angle(a=original)
for fraction in np.linspace(0, 1, 50):
    a = original.copy()
    a[-1] = fraction * original[-1]
    R = matrix_from_axis_angle(a)
    plot_basis(ax, R, alpha=0.2)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1.1)
ax.view_init(azim=105, elev=12)
plt.show()
