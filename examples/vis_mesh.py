"""
==============
Visualize Mesh
==============

This example shows how to load an STL mesh. This example must be
run from within the main folder because it uses a
hard-coded path to the STL file.
"""
import numpy as np
from pytransform3d import visualizer as pv


fig = pv.figure()
fig.plot_mesh(filename="test/test_data/cone.stl", s=5 * np.ones(3))
fig.plot_transform(A2B=np.eye(4), s=0.3)
fig.view_init()
fig.show()