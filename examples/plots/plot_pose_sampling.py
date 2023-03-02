"""
=============
Pose Sampling
=============
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt

mean = pt.transform_from(R=np.eye(3), p=np.array([0.0, 0.0, 0.5]))
cov = np.diag([0.001, 0.001, 0.5, 0.001, 0.001, 0.001])
rng = np.random.default_rng(0)
ax = None
for _ in range(200):
    pose = pt.random_transform(rng=rng, mean=mean, cov=cov)
    ax = pt.plot_transform(ax=ax, A2B=pose, s=0.3)
plt.show()
