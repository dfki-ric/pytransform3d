"""
=============
Plot Bivector
=============

Visualizes a bivector constructed from the wedge product of two vectors.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr


a = np.array([0.8, 0.3, 0.8])
b = np.array([0.4, 1.0, 0.3])

ax = pr.plot_basis()
pr.plot_bivector(ax=ax, a=a, b=b)
ax.view_init(elev=45, azim=75)
ax.set_xlim((0, 1))
ax.set_ylim((0, 1))
ax.set_zlim((0, 1))
plt.show()
