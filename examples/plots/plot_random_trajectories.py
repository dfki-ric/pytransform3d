"""
===================
Random Trajectories
===================

TODO
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr

n_trajectories = 3
start = np.eye(4)
goal = pt.transform_from(R=np.eye(3), p=0.3 * np.ones(3))
trajectories = ptr.random_trajectories(
    rng=np.random.default_rng(5),
    n_trajectories=n_trajectories,
    n_steps=1001,
    start=start,
    goal=goal,
    std_dev=np.array([200, 200, 200, 50, 50, 50])
)
plt.figure(figsize=(8, 8))
for i in range(n_trajectories):
    ax = plt.subplot(n_trajectories, 2, 1 + 2 * i, projection="3d")
    plt.setp(
        ax, xlim=(-0.1, 0.5), ylim=(-0.1, 0.5), zlim=(-0.1, 0.5),
        xlabel="X", ylabel="Y")
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
plt.tight_layout()
plt.show()
