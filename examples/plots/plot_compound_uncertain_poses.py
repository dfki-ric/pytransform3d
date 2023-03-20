"""
========================
Compound Uncertain Poses
========================

Each of the poses is has an associated covariance that is considered.

This example adapted and modified to 3D from

Barfoot, Furgale: Associating Uncertainty With Three-Dimensional Poses for Use
in Estimation Problems, http://ncfrn.mcgill.ca/members/pubs/barfoot_tro14.pdf
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.uncertainty as pu
import pytransform3d.plot_utils as ppu


rng = np.random.default_rng(0)
cov_pose_chol = np.diag([0.03, 0, 0.03, 0, 0, 0])
cov_pose = np.dot(cov_pose_chol, cov_pose_chol.T)
velocity_vector = np.array([0, 0, 0, 1.0, 0, 0])
T_vel = pt.transform_from_exponential_coordinates(velocity_vector)
n_steps = 100
n_mc_samples = 1000

T_est = np.eye(4)
path = np.zeros((n_steps + 1, 6))
path[0] = pt.exponential_coordinates_from_transform(T_est)
cov_est = cov_pose
for t in range(n_steps):
    T_est, cov_est = pu.concat_uncertain_transforms(
        T_est, cov_est, T_vel, cov_pose)
    path[t + 1] = pt.exponential_coordinates_from_transform(T_est)

T = np.eye(4)
mc_path = np.zeros((n_steps + 1, n_mc_samples, 4, 4))
mc_path[0, :] = T
for t in range(n_steps):
    diff_samples = ptr.transforms_from_exponential_coordinates(
        cov_pose_chol.dot(rng.standard_normal(size=(6, n_mc_samples))).T)
    for i in range(n_mc_samples):
        mc_path[t + 1, i] = diff_samples[i].dot(T_vel).dot(mc_path[t, i])
# Plot the random samples' trajectory lines (in a frame attached to the start)
mc_path_vec = np.zeros((n_steps, n_mc_samples, 3))
for t in range(n_steps):
    for i in range(n_mc_samples):
        mc_path_vec[t, i] = mc_path[t, i, :3, :3].T.dot(
            mc_path[t, i, :3, 3])

ax = ppu.make_3d_axis(100)

for i in range(mc_path_vec.shape[1]):
    ax.plot(
        mc_path_vec[:, i, 0], mc_path_vec[:, i, 1], mc_path_vec[:, i, 2],
        lw=1, c="b", alpha=0.05)
ax.scatter(
    mc_path_vec[-1, :, 0],
    mc_path_vec[-1, :, 1],
    mc_path_vec[-1, :, 2],
    s=5, c="b")

ax.plot(path[:, 3], path[:, 4], path[:, 5], lw=3, color="k")

pu.plot_projected_ellipsoid(ax, T_est, cov_est, color="g", factor=3.0)

mean_mc = np.mean(mc_path_vec[-1, :], axis=0)
cov_mc = np.cov(mc_path_vec[-1, :], rowvar=False)

factors = [1.65, 1.96, 2.58]
for factor in factors:
    ellipsoid2origin, radii = pu.to_ellipsoid(mean_mc, cov_mc)
    ppu.plot_ellipsoid(
        ax, factor * radii, ellipsoid2origin, wireframe=False, alpha=0.2,
        color="r")

plt.xlim((-5, 105))
plt.ylim((-50, 50))
plt.xlabel("x")
plt.ylabel("y")
ax.view_init(elev=70, azim=-90)
plt.show()
