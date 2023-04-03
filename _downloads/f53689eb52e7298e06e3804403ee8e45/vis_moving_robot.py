"""
==============
Animated Robot
==============

In this example we animate a 6-DOF robot arm with cylindrical visuals.
"""
import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv


def animation_callback(step, n_frames, tm, graph, joint_names):
    angle = 0.5 * np.cos(2.0 * np.pi * (step / n_frames))
    for joint_name in joint_names:
        tm.set_joint(joint_name, angle)
    graph.set_data()
    return graph


BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "pytransform3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)
joint_names = ["joint%d" % i for i in range(1, 7)]
for joint_name in joint_names:
    tm.set_joint(joint_name, 0.5)

fig = pv.figure()
graph = fig.plot_graph(
    tm, "robot_arm", s=0.1, show_frames=True, show_visuals=True)
fig.view_init()
fig.set_zoom(1.5)
n_frames = 100
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True,
                fargs=(n_frames, tm, graph, joint_names))
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
