"""
==============================
Compute Rotor from Two Vectors
==============================

We compute a rotor that rotates one vector to another vector.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
from pytransform3d.plot_utils import plot_vector


a = np.array([0.8, 0.3, 0.8])
b = np.array([0.4, 1.0, 0.1])
rotor = pr.rotor_from_two_vectors(a, b)
a_rotated = pr.rotor_apply(rotor, a)

ax = pr.plot_basis()
plot_vector(ax=ax, start=np.zeros(3), direction=a, color="#ff0000", alpha=0.8)
plot_vector(ax=ax, start=np.zeros(3), direction=b, color="#00ff00", alpha=0.8)
plot_vector(ax=ax, start=np.zeros(3), direction=a_rotated, color="#0000ff", alpha=0.8)
pr.plot_wedge(ax=ax, a=a, b=b)
ax.view_init(azim=75)
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_zlim((0, 1))
plt.show()
