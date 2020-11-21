"""
================
URDF with Meshes
================

This example shows how to load a URDF with STL meshes. This example must be
run from within the examples folder or the main folder because it uses a
hard-coded path to the URDF file and the meshes.
"""
import os
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv

BASE_DIR = "test/test_data/"
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.join("..", BASE_DIR)

tm = UrdfTransformManager()
with open(BASE_DIR + "simple_mechanism.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path=BASE_DIR)
tm.set_joint("joint", -1.1)

fig = pv.figure("URDF with meshes")
fig.plot_graph(tm, "lower_cone", s=0.1, show_frames=True,
               whitelist=["upper_cone", "lower_cone"],
               show_connections=True, show_visuals=True, show_name=False)
fig.view_init()
fig.show()
