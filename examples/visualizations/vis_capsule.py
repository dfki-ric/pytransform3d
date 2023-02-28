"""
=================
Visualize Capsule
=================
"""
import numpy as np
import pytransform3d.visualizer as pv


fig = pv.figure()
fig.plot_capsule(height=0.5, radius=0.1, c=(1, 0, 0))
fig.plot_transform(A2B=np.eye(4))
fig.view_init()
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
