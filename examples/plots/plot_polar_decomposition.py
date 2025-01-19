"""
========================
Plot Polar Decomposition
========================

Polar decomposition orthonormalizes basis vectors (i.e., rotation matrices).
It is more expensive than standard Gram-Schmidt orthonormalization, but it
spreads the error more evenly over all basis vectors. The top row of these
plots shows the unnormalized bases that were obtained by randomly rotating
one of the columns of the identity matrix. The middle row shows Gram-Schmidt
orthonormalization and the bottom row shows orthonormalization through polar
decomposition. For comparison, we show the unnormalized basis with dashed lines
in the last two rows.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr


n_cases = 4
plt.figure(figsize=(8, 8))
axes = [plt.subplot(3, n_cases, 1 + i, projection="3d")
        for i in range(3 * n_cases)]
ax_s = 1.0
plot_center = np.array([-0.2, -0.2, -0.2])
for ax in axes:
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    ax.set_xlim((-ax_s, ax_s))
    ax.set_ylim((-ax_s, ax_s))
    ax.set_zlim((-ax_s, ax_s))
axes[0].set_title("Unnormalized Bases")
axes[n_cases].set_title("Gram-Schmidt")
axes[2 * n_cases].set_title("Polar Decomposition")

rng = np.random.default_rng(46)
for i in range(n_cases):
    random_axis = rng.integers(0, 3)
    R_unnormalized = np.eye(3)
    R_unnormalized[:, random_axis] = np.dot(
        pr.random_matrix(rng, cov=0.1 * np.eye(3)),
        R_unnormalized[:, random_axis])
    pr.plot_basis(axes[i], R_unnormalized, p=plot_center, strict_check=False)

    R_gs = pr.norm_matrix(R_unnormalized)
    pr.plot_basis(axes[i + n_cases], R_unnormalized, p=plot_center,
                  strict_check=False, ls="--")
    pr.plot_basis(axes[i + n_cases], R_gs, p=plot_center)

    R_pd = pr.polar_decomposition(R_unnormalized)
    pr.plot_basis(axes[i + 2 * n_cases], R_unnormalized, p=plot_center,
                  strict_check=False, ls="--")
    pr.plot_basis(axes[i + 2 * n_cases], R_pd, p=plot_center)

plt.tight_layout()
plt.show()
