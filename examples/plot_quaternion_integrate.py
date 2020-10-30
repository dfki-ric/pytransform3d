"""
======================
Quaternion Integration
======================

Integrate angular velocities to a sequence of quaternions.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import quaternion_integrate, quaternion_gradient, matrix_from_quaternion, plot_basis


angular_velocities = np.zeros((21, 3))
angular_velocities[:, 1] = np.pi

Q = quaternion_integrate(angular_velocities, dt=0.1)
ax = None
for t in range(len(Q)):
    R = matrix_from_quaternion(Q[t])
    p = 2 * (t / (len(Q) - 1) - 0.5) * np.ones(3)
    ax = plot_basis(ax=ax, s=0.15, R=R, p=p)
plt.show()
