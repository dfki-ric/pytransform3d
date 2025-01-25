"""
===================
Random Trajectories
===================

TODO
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.trajectories as ptr


n_trajectories = 3
ax_s = 0.5
trajectories = ptr.random_trajectories(
    rng=np.random.default_rng(5),
    n_trajectories=n_trajectories,
    n_steps=1001,
    dt=0.01
)
plt.figure(figsize=(8, 8))
for i in range(n_trajectories):
    ax = plt.subplot(n_trajectories, 2, 1 + 2 * i, projection="3d")
    plt.setp(
        ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
        xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())
    ptr.plot_trajectory(
        ax=ax, P=ptr.pqs_from_transforms(trajectories[i]), s=0.1)
    ax = plt.subplot(n_trajectories, 2, 2 + 2 * i)
    for d in range(3):
        plt.plot(trajectories[i, :, d, 3].T, label="XYZ"[d])
    if i != n_trajectories - 1:
        ax.set_xticks(())
    else:
        ax.set_xlabel("Time step")
        ax.legend(loc="best")
    ax.set_ylabel("Position")
    ax.set_xlim((0, trajectories.shape[1] - 1))
plt.show()
