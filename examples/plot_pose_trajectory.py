"""
===============
Pose Trajectory
===============

Plotting pose trajectories with pytransform is easy.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import q_id, quaternion_slerp
from pytransform3d.trajectories import plot_trajectory


n_steps = 100
P = np.empty((n_steps, 7))
P[:, 0] = np.linspace(-0.8, 0.8, n_steps) ** 2
P[:, 1] = np.linspace(-0.8, 0.8, n_steps) ** 2
P[:, 2] = np.linspace(-0.8, 0.8, n_steps)
q_end = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
P[:, 3:] = np.vstack([quaternion_slerp(q_id, q_end, t)
                      for t in np.linspace(0, 1, n_steps)])

plot_trajectory(P=P, s=0.1, lw=1, c="k")
plt.show()
