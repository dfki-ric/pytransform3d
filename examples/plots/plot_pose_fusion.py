"""
====
TODO
====
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.uncertainty as pu
import pytransform3d.transformations as pt


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
x_est = pt.exponential_coordinates_from_transform(T_est)

_, axes = plt.subplots(
    nrows=6, ncols=6, sharex=True, sharey=True, squeeze=True, figsize=(10, 10))
factors = [1.0, 1.65, 1.96]
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
                color=color, alpha=0.1, factors=factors)

        pu.plot_error_ellipse(
            ax, x_est[indices], cov_est[indices][:, indices],
            color="k", factors=factors)

        ax.set_xlim((-10, 10))
        ax.set_ylim((-10, 10))

plt.show()
