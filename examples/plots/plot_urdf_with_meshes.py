"""
================
URDF with Meshes
================

This example shows how to load a URDF with STL meshes. This example must be
run from within the examples folder or the main folder because it uses a
hard-coded path to the URDF file and the meshes.
"""
import os
import matplotlib.pyplot as plt
from pytransform3d.urdf import UrdfTransformManager


BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "pytransform3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
with open(os.path.join(data_dir, "simple_mechanism.urdf"), "r") as f:
    tm.load_urdf(f.read(), mesh_path=data_dir)
tm.set_joint("joint", -1.1)
ax = tm.plot_frames_in(
    "lower_cone", s=0.1, whitelist=["upper_cone", "lower_cone"],
    show_name=True)
ax = tm.plot_connections_in("lower_cone", ax=ax)
tm.plot_visuals("lower_cone", ax=ax)
ax.set_xlim((-0.1, 0.15))
ax.set_ylim((-0.1, 0.15))
ax.set_zlim((0.0, 0.25))
plt.show()
