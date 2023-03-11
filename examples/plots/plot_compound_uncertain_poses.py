"""
========================
Compound Uncertain Poses
========================

Each of the poses is has an associated covariance that is considered.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.uncertainty as pu
import pytransform3d.transformations as pt


rng = np.random.default_rng(0)
cov_pose = np.diag([1e-5, 1e-5, 1e-3, 1e-5, 0.001, 0.001])
velocity = np.array([1.0, 0.0, 0.0])
n_steps = 100
n_mc_samples = 100
plot_dimensions = np.array([3, 4], dtype=int)

cov_chol = np.linalg.cholesky(cov_pose)

T = np.eye(4)
path = np.zeros((n_steps + 1, len(plot_dimensions)))
path[0] = pu.vector_from_transform(T)[plot_dimensions]
mc_path = np.zeros((n_steps + 1, n_mc_samples, 4, 4))
mc_path[0, :] = T
cov = cov_pose
e = np.zeros(6)
for t in range(n_steps):
    T_diff = pt.transform_from(R=np.eye(3), p=velocity)
    T, cov = pu.compund_poses(T_diff, cov_pose, T, cov)
    e = pu.vector_from_transform(T)
    path[t + 1] = e[plot_dimensions]

    diff_samples = np.hstack((np.zeros(3), velocity))[np.newaxis] + cov_chol.dot(rng.standard_normal(size=(6, n_mc_samples))).T
    for i in range(n_mc_samples):
        mc_path[t + 1, i] = pu.transform_from_vector(diff_samples[i]).dot(mc_path[t, i])

mc_path_vec = np.zeros((n_steps, n_mc_samples, 6))
for t in range(n_steps):
    for i in range(n_mc_samples):
        mc_path_vec[t, i] = pu.vector_from_transform(mc_path[t, i])

plt.plot(mc_path_vec[:, :, plot_dimensions[0]], mc_path_vec[:, :, plot_dimensions[1]], lw=1, c="b", alpha=0.5)
plt.plot(path[:, 0], path[:, 1], lw=3, color="k")

pu.plot_error_ellipse(
    plt.gca(), e[plot_dimensions], cov[plot_dimensions][:, plot_dimensions],
    color="b", alpha=0.4, factors=[1.65, 1.96, 2.58])

plt.show()
