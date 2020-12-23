"""
==============================
Visualize Cylinder with Wrench
==============================

We apply a constant body-fixed wrench to a cylinder and integrate
acceleration to twist and exponential coordinates of transformation
to finally compute the new pose of the cylinder.
"""
print(__doc__)


import numpy as np
from pytransform3d.transformations import exponential_coordinates_from_transform, transform_from_exponential_coordinates
import pytransform3d.visualizer as pv


def spatial_inertia_of_cylinder(mass, length, radius):
    I_xx = I_yy = 0.25 * mass * radius ** 2 + 1.0 / 12.0 * mass * length ** 2
    I_zz = 0.5 * mass * radius ** 2
    I = np.eye(6)
    I[:3, :3] *= np.array([I_xx, I_yy, I_zz])
    I[3:, 3:] *= mass
    return I


def animation_callback(
        step, cylinder, cylinder_frame, prev_cylinder2world,
        Stheta_dot, inertia_inv):
    if step == 0:  # Reset cylinder state
        prev_cylinder2world[:, :] = np.eye(4)
        Stheta_dot[:] = 0.0

    Stheta = exponential_coordinates_from_transform(prev_cylinder2world)

    # Apply constant wrench
    wrench_in_cylinder = np.array([0.1, 0.001, 0.001, 0.01, 1.0, 1.0])
    dt = 0.0005

    Stheta_ddot = np.dot(inertia_inv, wrench_in_cylinder)
    Stheta_dot += dt * Stheta_ddot
    Stheta += dt * Stheta_dot
    cylinder2world = transform_from_exponential_coordinates(Stheta)

    # Update visualization
    cylinder_frame.set_data(cylinder2world)
    cylinder.set_data(cylinder2world)

    prev_cylinder2world[:, :] = transform_from_exponential_coordinates(Stheta)

    return cylinder_frame, cylinder


fig = pv.figure()

# Definition of cylinder
mass = 1.0
length = 0.5
radius = 0.1
inertia_inv = np.linalg.inv(
    spatial_inertia_of_cylinder(mass=mass, length=length, radius=radius))

# State of cylinder
cylinder2world = np.eye(4)
twist = np.zeros(6)

cylinder = fig.plot_cylinder(length=length, radius=radius, c=[1, 0.5, 0])
cylinder_frame = fig.plot_transform(A2B=cylinder2world, s=0.5)

fig.plot_transform(A2B=np.eye(4), s=0.5)

fig.view_init()

fig.animate(
    animation_callback, n_frames=10000,
    fargs=(cylinder, cylinder_frame, cylinder2world, twist, inertia_inv),
    loop=True)

fig.show()
