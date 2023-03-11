"""
========================
Compound Uncertain Poses
========================

Each of the poses is has an associated covariance that is considered.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.uncertainty as pu


rng = np.random.default_rng(0)
cov_pose_chol = np.diag([0, 0, 0.03, 0, 0, 0])
cov_pose = np.dot(cov_pose_chol, cov_pose_chol.T)
velocity_vector = np.array([0, 0, 0, 1.0, 0, 0])
T_vel = pu.transform_from_vector(velocity_vector)
n_steps = 100
n_mc_samples = 100
plot_dimensions = np.array([3, 4], dtype=int)

T = np.eye(4)
path = np.zeros((n_steps + 1, 6))
path[0] = pu.vector_from_transform(T)
cov = cov_pose
for t in range(n_steps):
    T, cov = pu.compund_poses(T_vel, cov_pose, T, cov)
    path[t + 1] = pu.vector_from_transform(T)

T = np.eye(4)
mc_path = np.zeros((n_steps + 1, n_mc_samples, 4, 4))
mc_path[0, :] = T
for t in range(n_steps):
    diff_samples = cov_pose_chol.dot(rng.standard_normal(size=(6, n_mc_samples))).T
    for i in range(n_mc_samples):
        mc_path[t + 1, i] = pu.transform_from_vector(diff_samples[i]).dot(T_vel).dot(mc_path[t, i])
mc_path_vec = np.zeros((n_steps, n_mc_samples, 6))
for t in range(n_steps):
    for i in range(n_mc_samples):
        mc_path_vec[t, i] = pu.vector_from_transform(mc_path[t, i])

plt.plot(
    mc_path_vec[:, :, plot_dimensions[0]],
    mc_path_vec[:, :, plot_dimensions[1]], lw=1, c="b", alpha=0.1)
plt.scatter(
    mc_path_vec[-1, :, plot_dimensions[0]],
    mc_path_vec[-1, :, plot_dimensions[1]], s=5, c="b")
plt.plot(
    path[:, plot_dimensions[0]], path[:, plot_dimensions[1]], lw=3, color="k")

factors = [1.65, 1.96, 2.58]
pu.plot_error_ellipse(
    plt.gca(), path[-1, plot_dimensions],
    cov[plot_dimensions][:, plot_dimensions],
    color="b", alpha=0.4, factors=factors)

mean_mc = np.mean(mc_path_vec[-1, :, plot_dimensions], axis=1)
cov_mc = np.cov(mc_path_vec[-1, :, plot_dimensions], rowvar=True)

pu.plot_error_ellipse(
    plt.gca(), mean_mc, cov_mc, color="r", alpha=0.4, factors=factors)

plt.xlim((-5, 105))
plt.ylim((-45, 45))
plt.show()
