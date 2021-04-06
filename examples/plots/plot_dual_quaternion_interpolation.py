"""
=============================
Dual Quaternion Interpolation
=============================

This example shows interpolated trajectories between two random poses.
The red line corresponds to an interpolation with exponential coordinates
and the green line corresponds to an interpolation with dual quaternions.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.plot_utils as ppu


random_state = np.random.RandomState(21)
pose1 = pt.random_transform(random_state)
pose2 = pt.random_transform(random_state)
dq1 = pt.dual_quaternion_from_transform(pose1)
dq2 = -pt.dual_quaternion_from_transform(pose2)
Stheta1 = pt.exponential_coordinates_from_transform(pose1)
Stheta2 = pt.exponential_coordinates_from_transform(pose2)

n_steps = 100
interpolated_dqs = (np.linspace(1, 0, n_steps)[:, np.newaxis] * dq1 +
                    np.linspace(0, 1, n_steps)[:, np.newaxis] * dq2)
# renormalization (not required here because it will be done with conversion)
interpolated_dqs /= np.linalg.norm(
    interpolated_dqs[:, :4], axis=1)[:, np.newaxis]
interpolated_poses_from_dqs = np.array([
    pt.transform_from_dual_quaternion(dq) for dq in interpolated_dqs])
interpolated_ecs = (np.linspace(1, 0, n_steps)[:, np.newaxis] * Stheta1 +
                    np.linspace(0, 1, n_steps)[:, np.newaxis] * Stheta2)
interpolates_poses_from_ecs = ptr.transforms_from_exponential_coordinates(
    interpolated_ecs)

ax = pt.plot_transform(A2B=pose1, s=0.3, ax_s=2)
pt.plot_transform(A2B=pose2, s=0.3, ax=ax)
traj_from_dqs = ppu.Trajectory(interpolated_poses_from_dqs, s=0.1, c="g")
traj_from_dqs.add_trajectory(ax)
traj_from_ecs = ppu.Trajectory(interpolates_poses_from_ecs, s=0.1, c="r")
traj_from_ecs.add_trajectory(ax)
plt.show()
