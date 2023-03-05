"""
====
TODO
====
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pytransform3d.uncertainty as pu
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def fuse_poses_mc(means, covs, n_samples=10000, rng=np.random.default_rng(0)):
    """TODO"""
    n_poses = len(means)
    cov_L = [np.linalg.cholesky(cov) for cov in covs]
    pose_indices = rng.integers(0, n_poses, n_samples)
    epsilons = rng.standard_normal((n_samples, 6))
    samples = np.empty((n_samples, 6))
    for n in range(n_samples):
        samples[n] = pu.tran2vec(
            np.dot(pu.vec2tran(
                np.dot(cov_L[pose_indices[n]], epsilons[n])),
                means[pose_indices[n]]))
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=0, bias=True)
    return pu.vec2tran(mean), cov


x_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 6.0])
T_true = pu.vec2tran(x_true)
alpha = 5.0
cov1 = alpha * np.diag([0.1, 0.2, 0.1, 2.0, 1.0, 1.0])
cov2 = alpha * np.diag([0.1, 0.1, 0.2, 1.0, 3.0, 1.0])
cov3 = alpha * np.diag([0.2, 0.1, 0.1, 1.0, 1.0, 5.0])

rng = np.random.default_rng(0)

T1 = np.array([
    [0.8573, -0.2854, 0.4285, 3.5368],
    [-0.1113, 0.7098, 0.6956, -3.5165],
    [-0.5026, -0.6440, 0.5767, -0.9112],
    [0.0, 0.0, 0.0, 1.0000]
])
T1[:3, :3] = pr.norm_matrix(T1[:3, :3])
T2 = np.array([
    [0.5441, -0.6105, 0.5755, -1.0935],
    [0.8276, 0.5032, -0.2487, 5.5992],
    [-0.1377, 0.6116, 0.7791, 0.2690],
    [0.0, 0.0, 0.0, 1.0000]
])
T2[:3, :3] = pr.norm_matrix(T2[:3, :3])
T3 = np.array([
    [-0.0211, -0.7869, 0.6167, -3.0968],
    [-0.2293, 0.6042, 0.7631, 2.0868],
    [-0.9731, -0.1254, -0.1932, 2.0239],
    [0.0, 0.0, 0.0, 1.0000]
])
T3[:3, :3] = pr.norm_matrix(T3[:3, :3])

x1 = pu.tran2vec(T1)
x2 = pu.tran2vec(T2)
x3 = pu.tran2vec(T3)

T_est, cov_est = pu.fuse_poses([T1, T2, T3], [cov1, cov2, cov3])
T_mc, cov_mc = fuse_poses_mc([T1, T2, T3], [cov1, cov2, cov3])
print(T_est)
print(T_mc)
print(np.round(cov_est, 2))
print(np.round(cov_mc, 2))
x_est = pu.tran2vec(T_est)
x_mc = pu.tran2vec(T_mc)

_, axes = plt.subplots(
    nrows=6, ncols=6, sharex=True, sharey=True, squeeze=True, figsize=(10, 10))
factors = [1.0]
for i in range(6):
    for j in range(6):
        if i == j:
            continue

        indices = np.array([i, j])
        ax = axes[i][j]

        ax.scatter(x_true[i], x_true[j])

        for x, cov, color in zip([x1, x2, x3], [cov1, cov2, cov3], "rgb"):
            pu.plot_error_ellipse(
                ax, x[indices], cov[indices][:, indices],
                color=color, alpha=0.4, factors=factors)

        pu.plot_error_ellipse(
            ax, x_est[indices], cov_est[indices][:, indices],
            color="k", alpha=0.4, factors=factors)

        angle, width, height = pu.to_ellipse(cov_mc[indices][:, indices], 1.0)
        ell = Ellipse(xy=x_mc[indices], width=2.0 * width, height=2.0 * height,
                      angle=np.degrees(angle), fill=False, edgecolor="k",
                      linewidth=2)
        ax.add_artist(ell)

        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))

plt.show()
