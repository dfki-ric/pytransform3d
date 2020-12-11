"""TODO"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform, plot_screw_motion, twist_from_screw_displacement, transform_from_twist_displacement, screw_displacement_from_twist


q = np.array([-np.sqrt(0.3), -np.sqrt(0.3), -np.sqrt(0.3)])
s_axis = np.array([0, 0, 1])
h = 0.2
theta = 5.0
#A2B = transform_from_twist_displacement(twist)
ax = plot_transform(s=0.2)
#plot_transform(ax=ax, A2B=A2B, s=0.2)
ax = plot_screw_motion(ax=ax, q=q, s_axis=s_axis, h=h, theta=theta, s=0.5)
plt.show()
