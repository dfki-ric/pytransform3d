"""
====
TODO
====
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import pytransform3d.uncertainty as pu
import pytransform3d.transformations as pt


def fuse_poses_mc(means, covs, n_samples=10000, rng=np.random.default_rng(0)):
    """TODO"""
    n_poses = len(means)
    cov_L = [np.linalg.cholesky(cov) for cov in covs]
    pose_indices = rng.integers(0, n_poses, n_samples)
    epsilons = rng.standard_normal((n_samples, 6))
    samples = np.empty((n_samples, 6))
    for n in range(n_samples):
        samples[n] = pt.exponential_coordinates_from_transform(
            np.dot(pt.transform_from_exponential_coordinates(
                np.dot(cov_L[pose_indices[n]], epsilons[n])),
                means[pose_indices[n]]))
    mean = np.mean(samples, axis=0)
    cov = np.cov(samples, rowvar=0, bias=True)
    return pt.transform_from_exponential_coordinates(mean), cov


x_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 6.0])
T_true = pt.transform_from_exponential_coordinates(x_true)
alpha = 5.0
cov1 = alpha * np.diag([2.0, 1.0, 1.0, 0.1, 0.2, 0.1])
cov2 = alpha * np.diag([1.0, 3.0, 1.0, 0.1, 0.1, 0.2])
cov3 = alpha * np.diag([1.0, 1.0, 5.0, 0.2, 0.1, 0.1])

rng = np.random.default_rng(0)
T1 = pt.random_transform(rng, T_true, cov1)
x1 = pt.exponential_coordinates_from_transform(T1)
T2 = pt.random_transform(rng, T_true, cov2)
x2 = pt.exponential_coordinates_from_transform(T2)
T3 = pt.random_transform(rng, T_true, cov3)
x3 = pt.exponential_coordinates_from_transform(T3)

T_est, cov_est = pu.fuse_poses([T1, T2, T3], [cov1, cov2, cov3])
T_mc, cov_mc = fuse_poses_mc([T1, T2, T3], [cov1, cov2, cov3])
print(T_est)
print(T_mc)
print(np.round(cov_est, 2))
print(np.round(cov_mc, 2))
x_est = pt.exponential_coordinates_from_transform(T_est)
x_mc = pt.exponential_coordinates_from_transform(T_mc)

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
