"""
==============
Visualize Mesh
==============

This example shows how to load an STL mesh. This example must be
run from within the main folder because it uses a
hard-coded path to the STL file. Press 'H' to print the viewer's
help message to stdout.
"""
import os
import numpy as np
from pytransform3d import visualizer as pv


BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "pytransform3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)


fig = pv.figure()
fig.plot_mesh(filename=os.path.join(data_dir, "scan.stl"), s=np.ones(3))
fig.plot_transform(A2B=np.eye(4), s=0.3)
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
