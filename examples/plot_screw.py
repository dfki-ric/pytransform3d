"""TODO"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform, plot_screw, screw_axis_from_screw_parameters, transform_from_exponential_coordinates


q = np.array([-0.2, -0.1, -0.5])
s_axis = np.array([0, 0, 1])
h = 0.1
theta = 2.5 * np.pi
Stheta = screw_axis_from_screw_parameters(q, s_axis, h, theta)
A2B = transform_from_exponential_coordinates(Stheta)
ax = plot_transform(s=0.4)
plot_transform(ax=ax, A2B=A2B, s=0.2)
ax = plot_screw(ax=ax, q=q, s_axis=s_axis, h=h, theta=theta)
plt.show()
