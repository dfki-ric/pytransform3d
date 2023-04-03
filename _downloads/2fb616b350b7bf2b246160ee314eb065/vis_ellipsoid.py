"""
===================
Visualize Ellipsoid
===================
"""
import numpy as np
import pytransform3d.visualizer as pv


fig = pv.figure()
fig.plot_ellipsoid(radii=[0.2, 1, 0.5], c=(0.5, 0.5, 0))
fig.plot_transform(A2B=np.eye(4))
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
