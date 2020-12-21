"""
==============
Animated Robot
==============

In this example we animate a 6-DOF robot arm with cylindrical visuals.
"""
print(__doc__)


import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv


def animation_callback(step, n_frames, tm, graph, joint_names):
    angle = 0.5 * np.sin(2.0 * np.pi * (step / n_frames))
    for joint_name in joint_names:
        tm.set_joint(joint_name, angle)
    graph.set_data()
    return graph

BASE_DIR = "test/test_data/"
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.join("..", BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(BASE_DIR, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=BASE_DIR)
joint_names = ["joint%d" % i for i in range(1, 7)]

fig = pv.figure()
graph = fig.plot_graph(
    tm, "robot_arm", s=0.1, show_frames=True, show_visuals=True)
fig.view_init()
n_frames = 100
fig.animate(animation_callback, n_frames, loop=True,
            fargs=(n_frames, tm, graph, joint_names))
fig.show()
