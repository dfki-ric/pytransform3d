"""TODO"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import active_matrix_from_extrinsic_roll_pitch_yaw
from pytransform3d.transformations import (plot_transform, plot_screw,
                                           screw_axis_from_screw_parameters,
                                           transform_from_exponential_coordinates,
                                           concat, transform_from)


q = np.array([-0.2, -0.1, -0.5])
s_axis = np.array([0, 0, 1])
h = 0.05
theta = 5.5 * np.pi
Stheta = screw_axis_from_screw_parameters(q, s_axis, h, theta)
A2B = transform_from_exponential_coordinates(Stheta)
origin = transform_from(active_matrix_from_extrinsic_roll_pitch_yaw([0.5, -0.3, 0.2]), np.array([0.0, 0.1, 0.1]))
ax = plot_transform(A2B=origin, s=0.4)
plot_transform(ax=ax, A2B=concat(A2B, origin), s=0.2)
plot_screw(ax=ax, q=q, s_axis=s_axis, h=h, theta=theta, A2B=origin, s=1.5, alpha=0.6)
ax.view_init(elev=40, azim=170)
plt.show()
