"""
================
Visualize Sphere
================
"""
import numpy as np
import pytransform3d.visualizer as pv


fig = pv.figure()
fig.plot_sphere(radius=0.5)
fig.plot_transform(A2B=np.eye(4))
fig.show()