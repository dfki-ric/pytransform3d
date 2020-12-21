"""TODO"""
print(__doc__)


import os
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv


BASE_DIR = "test/test_data/"
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.join("..", BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(BASE_DIR, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=BASE_DIR)
tm.set_joint("joint2", 0.2 * np.pi)
tm.set_joint("joint3", 0.2 * np.pi)
tm.set_joint("joint5", 0.2 * np.pi)
tm.set_joint("joint6", 0.5 * np.pi)

ee2base = tm.get_transform("tcp", "robot_arm")

fig = pv.figure()
fig.plot_transform(s=0.4)
fig.plot_transform(A2B=ee2base, s=0.1)
graph = fig.plot_graph(tm, "robot_arm", s=0.1, show_visuals=True)
fig.view_init()
fig.show()
