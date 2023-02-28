"""
===============
Visualize Plane
===============

Visualizes one plane in Hesse normal form and one plane defined by point and
normal.
"""
import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.visualizer as pv


fig = pv.figure()
fig.plot_transform(np.eye(4))
rng = np.random.default_rng(8853)
fig.plot_plane(pr.norm_vector(rng.standard_normal(3)), rng.standard_normal(),
               c=(1, 0.5, 0))
fig.plot_plane(pr.norm_vector(rng.standard_normal(3)),
               point_in_plane=rng.standard_normal(3), c=(0, 1, 1))
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
