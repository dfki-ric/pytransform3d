"""
============================
Inverse of an Uncertain Pose
============================
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.uncertainty as pu
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr


A2B = pt.transform_from(
    R=pr.matrix_from_euler([1.5, 0.5, 1.3], 0, 1, 2, True),
    p=[5.0, 1.0, 2.0]
)
alpha = 2.0
cov_A2B = alpha * np.diag([0.1, 0.2, 0.1, 2.0, 1.0, 1.0])
x_A2B = pt.exponential_coordinates_from_transform(A2B)
B2A, cov_B2A = pu.invert_uncertain_transform(A2B, cov_A2B)
x_B2A = pt.exponential_coordinates_from_transform(B2A)

n_samples = 100
rng = np.random.default_rng(1)
samples_A2B = ptr.exponential_coordinates_from_transforms(
    [pt.random_transform(rng, A2B, cov_A2B) for _ in range(n_samples)])
samples_B2A = ptr.exponential_coordinates_from_transforms(
    [pt.random_transform(rng, B2A, cov_B2A) for _ in range(n_samples)])

factors = [1.65, 1.96, 2.58]
_, axes = plt.subplots(
    nrows=6, ncols=6, sharex=True, sharey=True, squeeze=True, figsize=(10, 10))
for i in range(6):
    for j in range(6):
        if i == j:
            continue

        indices = np.array([i, j])
        ax = axes[i][j]

        ax.scatter(
            samples_A2B[:, i], samples_A2B[:, j], color="g", s=5, alpha=0.3)
        pu.plot_error_ellipse(
            ax, x_A2B[indices], cov_A2B[indices][:, indices],
            color="g", alpha=0.3, factors=factors)

        ax.scatter(
            samples_B2A[:, i], samples_B2A[:, j], color="r", s=5, alpha=0.3)
        pu.plot_error_ellipse(
            ax, x_B2A[indices], cov_B2A[indices][:, indices],
            color="r", alpha=0.3, factors=factors)

        ax.scatter(0, 0, color="k")

plt.show()
