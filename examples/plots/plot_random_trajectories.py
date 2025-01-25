"""
===================
Random Trajectories
===================

TODO
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.trajectories as ptr


trajectories = ptr.random_trajectories(
    rng=np.random.default_rng(5),
    n_trajectories=3,
    n_steps=1001,
    dt=0.01
)
ax = plt.subplot(4, 1, 1, projection="3d")
for trajectory in trajectories:
    ptr.plot_trajectory(ax=ax, P=ptr.pqs_from_transforms(trajectory))
for i in range(3):
    ax = plt.subplot(4, 1, 2 + i)
    if i != 2:
        ax.set_xticks(())
    ax.set_xlim((0, trajectories.shape[1] - 1))
    plt.plot(trajectories[:, :, i, 3].T)
plt.show()
