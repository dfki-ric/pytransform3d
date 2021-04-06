"""
===========
Plot Sphere
===========
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import plot_sphere, remove_frame


random_state = np.random.RandomState(42)
ax = plot_sphere(
    radius=0.5, wireframe=False, alpha=0.1, color="k", n_steps=20, ax_s=0.5)
plot_sphere(ax=ax, radius=0.5, wireframe=True)
plot_transform(ax=ax, A2B=np.eye(4), s=0.3, lw=3)
remove_frame(ax)
plt.show()
