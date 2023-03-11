"""
============
Fuse 3 Poses
============

Each of the poses is has an associated covariance that is considered during
the fusion. Each of the plots shows a projection of the 6D pose vector to
two dimensions.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.uncertainty as pu
import pytransform3d.transformations as pt


x_true = np.array([1.0, 0.0, 0.0, 0.0, 0.0, np.pi / 6.0])
T_true = pu.transform_from_vector(x_true)
alpha = 5.0
cov1 = alpha * np.diag([0.1, 0.2, 0.1, 2.0, 1.0, 1.0])
cov2 = alpha * np.diag([0.1, 0.1, 0.2, 1.0, 3.0, 1.0])
cov3 = alpha * np.diag([0.2, 0.1, 0.1, 1.0, 1.0, 5.0])

rng = np.random.default_rng(0)

T1 = np.dot(pu.transform_from_vector(pt.random_exponential_coordinates(rng=rng, cov=cov1)), T_true)
T2 = np.dot(pu.transform_from_vector(pt.random_exponential_coordinates(rng=rng, cov=cov2)), T_true)
T3 = np.dot(pu.transform_from_vector(pt.random_exponential_coordinates(rng=rng, cov=cov3)), T_true)
T4 = np.dot(pu.transform_from_vector(pt.random_exponential_coordinates(rng=rng, cov=cov3)), T_true)

x1 = pu.vector_from_transform(T1)
x2 = pu.vector_from_transform(T2)
x3 = pu.vector_from_transform(T3)
x4 = pu.vector_from_transform(T4)

T_est, cov_est, V = pu.fuse_poses([T1, T2, T3, T4], [cov1, cov2, cov3, cov3])
x_est = pu.vector_from_transform(T_est)

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

        for x, cov, color in zip([x1, x2, x3, x4], [cov1, cov2, cov3, cov3], "rgbc"):
            pu.plot_error_ellipse(
                ax, x[indices], cov[indices][:, indices],
                color=color, alpha=0.4, factors=factors)

        pu.plot_error_ellipse(
            ax, x_est[indices], cov_est[indices][:, indices],
            color="k", alpha=0.4, factors=factors)

        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))

plt.show()
