"""
=================================
Rigid Body Dynamics Under Wrench
=================================

A constant body-fixed wrench is applied to a cylinder. At each time step we
integrate the resulting spatial acceleration to get the body twist and then
update the pose using exponential coordinates. The trajectory of the
cylinder's origin is drawn in real time as the simulation progresses.

The rigid-body equations of motion in body coordinates are::

    G * Sdot = W - ad(S)^T * G * S

where G is the 6x6 spatial inertia matrix, S is the body twist (screw
velocity), W is the wrench expressed in body coordinates, and ad(S)^T is the
adjoint map used to account for Coriolis effects. Here we neglect Coriolis
forces for simplicity and integrate directly with::

    Sdot = G^-1 * W

The pose update uses the matrix exponential::

    body2world(t+dt) = exp(dt * S) * body2world(t)
"""

import numpy as np

import pytransform3d.viser as pv
from pytransform3d.transformations import transform_from_exponential_coordinates


def spatial_inertia_of_cylinder(mass, length, radius):
    """Compute the 6x6 spatial inertia matrix of a uniform cylinder.

    The cylinder axis is aligned with the z-axis of its body frame.

    Parameters
    ----------
    mass : float
        Total mass [kg].

    length : float
        Length along z-axis [m].

    radius : float
        Radius [m].

    Returns
    -------
    G : array, shape (6, 6)
        Spatial inertia matrix (rotation part first, then translation).
    """
    I_xx = I_yy = 0.25 * mass * radius**2 + (1.0 / 12.0) * mass * length**2
    I_zz = 0.5 * mass * radius**2
    G = np.eye(6)
    G[:3, :3] *= np.array([I_xx, I_yy, I_zz])
    G[3:, 3:] *= mass
    return G


# %%
# Cylinder parameters and initial state
# ---------------------------------------

mass = 1.0
length = 0.5
radius = 0.1
G_inv = np.linalg.inv(spatial_inertia_of_cylinder(mass, length, radius))

# Body-fixed wrench: small torques about x/y and a force along y.
# This combination causes a tumbling corkscrew motion.
wrench_in_body = np.array([0.08, 0.04, 0.005, 0.0, 0.8, 0.5])

dt = 0.001
n_frames = 5000
# We collect a fixed-length rolling window of positions for the trail.
trail_length = 300

# %%
# Scene setup
# -----------

fig = pv.figure()

body2world = np.eye(4)
twist = np.zeros(6)

cylinder = fig.plot_cylinder(length=length, radius=radius, c=(0.9, 0.55, 0.1))
body_frame = fig.plot_transform(A2B=body2world, s=0.25)
world_frame = fig.plot_transform(A2B=np.eye(4), s=0.4)

# Trail line: start with two coincident points at the origin.
trail_positions = [np.zeros(3), np.zeros(3)]
trail_line = pv.Line3D(P=np.array(trail_positions), c=(0.2, 0.6, 1.0))
trail_line.add_artist(fig)


# %%
# Simulation and animation callback
# -----------------------------------
def animation_callback(step, cylinder, body_frame, trail_line, state):
    """Advance the simulation by one frame and update the scene.

    One animation frame covers multiple integration steps so the motion
    appears smooth even at 30 fps.

    Parameters
    ----------
    step : int
        Current animation frame index (unused, state is in ``state``).
    cylinder : Cylinder
        Cylinder artist.
    body_frame : Frame
        Coordinate frame artist attached to the cylinder.
    trail_line : Line3D
        Trail showing the history of the cylinder origin.
    state : list
        Mutable container ``[body2world, twist, trail_positions]``.

    Returns
    -------
    artists : tuple
        Updated artists.
    """
    body2world, twist, trail_positions = state

    # Integrate multiple steps per animation frame for smoother motion.
    steps_per_frame = 10
    for _ in range(steps_per_frame):
        twist += dt * G_inv.dot(wrench_in_body)
        new_pose = transform_from_exponential_coordinates(dt * twist).dot(
            body2world
        )
        body2world[:] = new_pose

    # Update geometry.
    cylinder.set_data(body2world)
    body_frame.set_data(body2world)

    # Append current position to the rolling trail.
    trail_positions.append(body2world[:3, 3].copy())
    if len(trail_positions) > trail_length:
        trail_positions.pop(0)
    trail_line.set_data(np.array(trail_positions))

    state[0] = body2world
    state[2] = trail_positions
    return cylinder, body_frame, trail_line


state = [body2world, twist, trail_positions]

if "__file__" in globals():
    fig.show()
    fig.animate(
        animation_callback,
        n_frames,
        loop=True,
        fargs=(cylinder, body_frame, trail_line, state),
    )
