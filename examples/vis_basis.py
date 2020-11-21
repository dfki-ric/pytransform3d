"""
==========================
Visualize Coordinate Frame
==========================
"""
import numpy as np
import pytransform3d.visualizer as pv


fig = pv.figure()
fig.plot_basis(R=np.eye(3), p=[0.1, 0.2, 0.3])
fig.view_init(azim=15, elev=30)
fig.show()