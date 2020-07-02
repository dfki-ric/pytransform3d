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
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.join("..", BASE_DIR)

tm = UrdfTransformManager()
with open(BASE_DIR + "simple_mechanism.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path=BASE_DIR)
tm.set_joint("joint", -1.1)
ax = tm.plot_frames_in(
    "lower_cone", s=0.1, whitelist=["upper_cone", "lower_cone"], show_name=True)
ax = tm.plot_connections_in("lower_cone", ax=ax)
tm.plot_visuals("lower_cone", ax=ax)
ax.set_xlim((-0.2, 0.2))
ax.set_ylim((-0.2, 0.2))
ax.set_zlim((0.0, 0.4))
plt.show()
