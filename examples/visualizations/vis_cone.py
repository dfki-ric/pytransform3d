"""
==============
Visualize Cone
==============
"""
import numpy as np
import pytransform3d.visualizer as pv


fig = pv.figure()
fig.plot_cone(height=1.0, radius=0.3, c=(0, 0, 0.5))
fig.plot_transform(A2B=np.eye(4))
fig.view_init()
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
