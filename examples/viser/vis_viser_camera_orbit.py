"""
======================
Camera Orbiting Scene
======================

A pinhole camera is animated orbiting around a static scene. The camera
frustum updates at each frame to show the camera's current position and
field of view.

The scene contains a coordinate frame at the origin, a box, a sphere, and
a cylinder. The camera position is constrained to a circle at a fixed radius
and height, always pointing toward the origin. The orbit plane is tilted
slightly so the camera looks at the scene from above.

Camera intrinsics
-----------------
The intrinsic matrix encodes the focal length and the principal point::

    M = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

The visualized frustum shows the four rays from the camera centre through the
image corners and a rectangle at ``virtual_image_distance`` from the camera.
"""

import numpy as np

import pytransform3d.viser as pv
from pytransform3d.transformations import transform_from

# %%
# Camera intrinsics
# -----------------

fl = 600.0  # focal length [pixels]
w, h = 640, 480  # sensor size [pixels]
M = np.array([[fl, 0, w / 2.0], [0, fl, h / 2.0], [0, 0, 1]], dtype=float)
virtual_image_distance = 0.6

# %%
# Static scene
# ------------

fig = pv.figure()
# Slightly above and to the side to see the full orbit arc.
fig.view_init(azim=45, elev=25, distance=4.5)

# World frame at the origin.
fig.plot_transform(A2B=np.eye(4), s=0.4)

# A few objects to look at.  Viser is y-up: [x, height, z].
fig.plot_box(
    size=[0.4, 0.3, 0.6],
    A2B=transform_from(np.eye(3), [0.3, 0.15, 0.0]),
    c=(0.7, 0.3, 0.2),
)
fig.plot_sphere(
    radius=0.18,
    A2B=transform_from(np.eye(3), [-0.35, 0.18, 0.25]),
    c=(0.2, 0.6, 0.85),
)
fig.plot_cylinder(
    length=0.55,
    radius=0.09,
    A2B=transform_from(np.eye(3), [0.0, 0.275, -0.4]),
    c=(0.3, 0.75, 0.3),
)

# %%
# Camera pose helper
# ------------------


def look_at(position, target=np.zeros(3)):
    """Build a cam2world transform that places the camera at *position*
    with its optical axis pointing toward *target*.

    The camera convention used here is z-forward, y-down.  Viser uses a
    y-up world coordinate system, so world-up is +y.

    Parameters
    ----------
    position : array, shape (3,)
        Camera position in world coordinates.

    target : array, shape (3,)
        Point the camera looks at.

    Returns
    -------
    cam2world : array, shape (4, 4)
        Transformation from camera frame to world frame.
    """
    z_cam = target - position  # forward
    z_cam /= np.linalg.norm(z_cam)
    # Viser is y-up; fall back to +z when looking straight up or down.
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_cam, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(z_cam, world_up)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    R = np.column_stack([x_cam, y_cam, z_cam])
    return transform_from(R, position)


# %%
# Initial camera pose and artist
# --------------------------------

orbit_radius = 1.8
orbit_height = 0.8
n_frames = 120

initial_cam2world = look_at(np.array([orbit_radius, orbit_height, 0.0]))
camera_artist = fig.plot_camera(
    M=M,
    cam2world=initial_cam2world,
    virtual_image_distance=virtual_image_distance,
    sensor_size=(w, h),
)
# Show the camera's own coordinate frame.
camera_frame = fig.plot_transform(A2B=initial_cam2world, s=0.18)


# %%
# Animation callback
# ------------------


def animation_callback(step, n_frames, camera_artist, camera_frame):
    """Move the camera one step along its circular orbit.

    Parameters
    ----------
    step : int
        Current frame index.
    n_frames : int
        Total number of frames in one orbit.
    camera_artist : Camera
        Camera frustum artist.
    camera_frame : Frame
        Coordinate frame artist for the camera.

    Returns
    -------
    artists : tuple
        Updated artists.
    """
    angle = 2.0 * np.pi * step / n_frames
    pos = np.array(
        [
            orbit_radius * np.cos(angle),
            orbit_height,
            orbit_radius * np.sin(angle),
        ]
    )
    cam2world = look_at(pos)
    camera_artist.set_data(cam2world=cam2world)
    camera_frame.set_data(cam2world)
    return camera_artist, camera_frame


if "__file__" in globals():
    fig.show()
    fig.animate(
        animation_callback,
        n_frames,
        loop=True,
        fargs=(n_frames, camera_artist, camera_frame),
    )
else:
    fig.save_image("__viser_rendered_image.jpg")
