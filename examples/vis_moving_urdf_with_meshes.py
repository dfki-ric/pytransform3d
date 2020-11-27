"""
=========================
Animated URDF with Meshes
=========================

This example shows how to load a URDF with STL meshes and animate it.
This example must be run from within the examples folder or the main
folder because it uses a hard-coded path to the URDF file and the meshes.
"""
print(__doc__)


import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv


def animation_callback(step, n_frames, tm, graph):
    angle = 2.79253 * np.sin(2.0 * np.pi * (step / n_frames))
    tm.set_joint("joint", angle)
    graph.set_data()
    return graph

BASE_DIR = "test/test_data/"
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.join("..", BASE_DIR)

tm = UrdfTransformManager()
with open(BASE_DIR + "simple_mechanism.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path=BASE_DIR)

fig = pv.figure("URDF with meshes")
graph = fig.plot_graph(
    tm, "lower_cone", s=0.1, show_connections=True, show_visuals=True)
fig.view_init()
n_frames = 100
fig.animate(animation_callback, n_frames, loop=True, fargs=(n_frames, tm, graph))
fig.show()
