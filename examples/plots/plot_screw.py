"""
========================================
Plot Transformation through Screw Motion
========================================

A screw axis is represented by the parameters (q, s_axis, h). We can represent
any transformation with a screw axis and an additional parameter theta that
encodes the rotation angle and through h * theta the translation. Here we
visualize a screw axis and the transformation generated from a specific
theta.

The larger coordinate frame represents the origin of the transformation
and the smaller frame represents the transformed frame. The red point
indicates the position of q, which is a point on the screw axis. A straight
arrow shows the direction of the screw axis. The spiral path represents
a displacement of length theta along the screw axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import active_matrix_from_extrinsic_roll_pitch_yaw
from pytransform3d.transformations import (
    plot_transform, plot_screw, screw_axis_from_screw_parameters,
    transform_from_exponential_coordinates, concat, transform_from)


# Screw parameters
q = np.array([-0.2, -0.1, -0.5])
s_axis = np.array([0, 0, 1])
h = 0.05
theta = 5.5 * np.pi

Stheta = screw_axis_from_screw_parameters(q, s_axis, h) * theta
A2B = transform_from_exponential_coordinates(Stheta)

origin = transform_from(
    active_matrix_from_extrinsic_roll_pitch_yaw([0.5, -0.3, 0.2]),
    np.array([0.0, 0.1, 0.1]))

ax = plot_transform(A2B=origin, s=0.4)
plot_transform(ax=ax, A2B=concat(A2B, origin), s=0.2)
plot_screw(
    ax=ax, q=q, s_axis=s_axis, h=h, theta=theta, A2B=origin, s=1.5, alpha=0.6)
ax.view_init(elev=40, azim=170)
plt.subplots_adjust(0, 0, 1, 1)
plt.show()
