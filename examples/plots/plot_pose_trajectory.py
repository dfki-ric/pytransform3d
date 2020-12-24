"""
===============
Pose Trajectory
===============

Plotting pose trajectories with pytransform3d is easy.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import q_id, quaternion_slerp
from pytransform3d.trajectories import plot_trajectory


n_steps = 100
P = np.empty((n_steps, 7))
P[:, 0] = np.cos(np.linspace(-2 * np.pi, 2 * np.pi, n_steps))
P[:, 1] = np.sin(np.linspace(-2 * np.pi, 2 * np.pi, n_steps))
P[:, 2] = np.linspace(-1, 1, n_steps)
q_end = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)])
P[:, 3:] = np.vstack([quaternion_slerp(q_id, q_end, t)
                      for t in np.linspace(0, 1, n_steps)])

ax = plot_trajectory(P=P, s=0.3, lw=2, c="k")
plt.show()
