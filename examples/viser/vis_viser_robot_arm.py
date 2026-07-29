"""
====================================
Animated Robot with TCP Trajectory
====================================

A 6-DOF robot arm is animated while the trajectory of its tool-centre point
(TCP) is traced in real time. Each joint follows an independent sinusoidal
profile, producing a complex end-effector path.

The TCP path drawn so far is shown as an orange line. After one full cycle the
line resets and a new cycle begins.

This example must be run from within the ``examples/`` folder or the main
repository folder because it uses a relative path to the URDF file.
"""

import os

import numpy as np

import pytransform3d.viser as pv
from pytransform3d.urdf import UrdfTransformManager

# %%
# Data loading
# ------------

BASE_DIR = "test/test_data/"
data_dir = BASE_DIR
search_path = "."
while (
    not os.path.exists(data_dir)
    and os.path.dirname(search_path) != "pytransform3d"
):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "robot_with_visuals.urdf")
with open(filename, "r") as f:
    tm.load_urdf(f.read(), mesh_path=data_dir)

joint_names = ["joint%d" % i for i in range(1, 7)]
# Independent phase offsets give a more varied TCP path.
phases = np.linspace(0, np.pi, len(joint_names))
n_frames = 150

# %%
# Pre-compute the full TCP trajectory so we can draw it incrementally.
tcp_path = []
for step in range(n_frames):
    for i, joint_name in enumerate(joint_names):
        angle = 0.5 * np.cos(2.0 * np.pi * step / n_frames + phases[i])
        tm.set_joint(joint_name, angle)
    tcp_path.append(tm.get_transform("tcp", "robot_arm")[:3, 3].copy())
tcp_path = np.array(tcp_path)  # shape (n_frames, 3)

# Reset joints to initial pose.
for joint_name in joint_names:
    tm.set_joint(joint_name, 0.0)

# %%
# Scene setup
# -----------
fig = pv.figure()
fig.view_init(azim=30, elev=35, center=(0.0, 0.3, 0.0), distance=2.5)
graph = fig.plot_graph(
    tm, "robot_arm", s=0.05, show_frames=True, show_visuals=True
)

# Initialize the trajectory line with a degenerate two-point segment so the
# artist has a valid handle from the start.
init_pos = tcp_path[0]
trajectory_line = pv.Line3D(
    P=np.vstack([init_pos, init_pos]),
    c=(1.0, 0.5, 0.0),
)
trajectory_line.add_artist(fig)


# %%
# Animation callback
# ------------------
def animation_callback(
    step, n_frames, tm, graph, trajectory_line, tcp_path, phases
):
    """Update robot pose and extend the TCP trajectory line.

    Parameters
    ----------
    step : int
        Current frame index.
    n_frames : int
        Total number of frames in one cycle.
    tm : UrdfTransformManager
        Robot kinematics manager.
    graph : Graph
        Artist representing the robot in the scene.
    trajectory_line : Line3D
        Artist for the TCP trajectory.
    tcp_path : array, shape (n_frames, 3)
        Pre-computed TCP positions.
    phases : array, shape (n_joints,)
        Phase offsets for each joint.

    Returns
    -------
    artists : tuple
        Updated artists.
    """
    for i, joint_name in enumerate(joint_names):
        angle = 0.5 * np.cos(2.0 * np.pi * step / n_frames + phases[i])
        tm.set_joint(joint_name, angle)
    graph.set_data()

    # Build path from start of cycle to current step.
    # Always pass at least 2 points to keep the handle alive.
    n_pts = max(2, step + 1)
    trajectory_line.set_data(tcp_path[:n_pts])

    return graph, trajectory_line


if "__file__" in globals():
    fig.show()
    fig.animate(
        animation_callback,
        n_frames,
        loop=True,
        fargs=(n_frames, tm, graph, trajectory_line, tcp_path, phases),
    )
else:
    fig.save_image("__viser_rendered_image.jpg")
