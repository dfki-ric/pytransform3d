"""
====================
Geometric Primitives
====================

All geometric primitive shapes supported by the viser backend are arranged
in two rows in the x-z ground plane (viser uses y-up). The first row shows
sphere, box, cylinder, and capsule. The second row shows cone, ellipsoid,
plane, and a camera frustum. Each shape is accompanied by a small coordinate
frame at its origin.

This example can be used as a reference for the available shapes and their
constructor arguments.
"""

import numpy as np

import pytransform3d.viser as pv
from pytransform3d.transformations import transform_from

fig = pv.figure()

# Viser is y-up: shapes are laid out in the x-z ground plane.
# Row 0 is at z=0, row 1 is at z=spacing_z.
spacing_x = 1.8
spacing_z = 2.0

# Center of the 4 x 2 layout; distance chosen to frame all shapes.
fig.view_init(
    azim=30, elev=30, center=(1.5 * spacing_x, 0.0, 0.5 * spacing_z), distance=9.0
)

# %%
# Row 0: sphere, box, cylinder, capsule
# --------------------------------------

p = np.array([0 * spacing_x, 0.0, 0.0])
fig.plot_sphere(radius=0.4, A2B=transform_from(np.eye(3), p), c=(0.8, 0.2, 0.2))
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

p = np.array([1 * spacing_x, 0.0, 0.0])
fig.plot_box(
    size=[0.5, 0.7, 0.45], A2B=transform_from(np.eye(3), p), c=(0.2, 0.75, 0.2)
)
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

p = np.array([2 * spacing_x, 0.0, 0.0])
fig.plot_cylinder(
    length=0.7,
    radius=0.25,
    A2B=transform_from(np.eye(3), p),
    c=(0.2, 0.2, 0.85),
)
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

p = np.array([3 * spacing_x, 0.0, 0.0])
fig.plot_capsule(
    height=0.45,
    radius=0.25,
    A2B=transform_from(np.eye(3), p),
    c=(0.8, 0.8, 0.15),
)
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

# %%
# Row 1: cone, ellipsoid, plane, camera frustum
# -----------------------------------------------

p = np.array([0 * spacing_x, 0.0, spacing_z])
fig.plot_cone(
    height=0.7, radius=0.3, A2B=transform_from(np.eye(3), p), c=(0.85, 0.4, 0.1)
)
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

p = np.array([1 * spacing_x, 0.0, spacing_z])
fig.plot_ellipsoid(
    radii=[0.5, 0.28, 0.18],
    A2B=transform_from(np.eye(3), p),
    c=(0.55, 0.1, 0.85),
)
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

p = np.array([2 * spacing_x, 0.0, spacing_z])
fig.plot_plane(normal=[0, 1, 0], point_in_plane=p, s=0.6, c=(0.1, 0.75, 0.75))
fig.plot_transform(A2B=transform_from(np.eye(3), p), s=0.35)

# %%
# Camera frustum: a pinhole camera tilted slightly downward.
# M is the intrinsic matrix for a 640x480 image at focal length 500 px.
fl = 500.0
w, h = 640, 480
M = np.array([[fl, 0, w / 2.0], [0, fl, h / 2.0], [0, 0, 1]], dtype=float)
# Tilt the camera 30 degrees around x so the frustum is easy to see.
tilt = np.pi / 6.0
R_tilt = np.array(
    [
        [1, 0, 0],
        [0, np.cos(tilt), -np.sin(tilt)],
        [0, np.sin(tilt), np.cos(tilt)],
    ]
)
cam2world = transform_from(R_tilt, [3 * spacing_x, 0.4, spacing_z])
fig.plot_camera(
    M=M, cam2world=cam2world, virtual_image_distance=0.55, sensor_size=(w, h)
)
fig.plot_transform(A2B=cam2world, s=0.35)

if "__file__" in globals():
    fig.show()
    input("Press Enter to exit...")
else:
    fig.save_image("__viser_rendered_image.jpg")
