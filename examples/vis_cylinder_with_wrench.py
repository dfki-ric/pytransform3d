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


"""
def adjoint_A2B(A2B):
    from pytransform3d.rotations import cross_product_matrix
    R = A2B[:3, :3]
    p = A2B[:3, 3]
    adj = np.zeros((6, 6))
    adj[:3, :3] = R
    adj[3:, :3] = np.dot(cross_product_matrix(p), R)
    adj[3:, 3:] = R
    return adj
"""


def animation_callback(step, n_frames, cylinder, cylinder_frame, prev_cylinder2world, inertia_inv, Stheta_dot):
    if step == 0:
        prev_cylinder2world[:, :] = np.eye(4)
        Stheta_dot[:] = 0.0

    wrench_in_cylinder = np.array([0.1, 0.001, 0.001, 0.01, 1.0, 1.0])
    dt = 0.0005

    Stheta_ddot = np.dot(inertia_inv, wrench_in_cylinder)
    Stheta_dot += dt * Stheta_ddot
    Stheta = exponential_coordinates_from_transform(prev_cylinder2world)
    Stheta += dt * Stheta_dot
    cylinder2world = transform_from_exponential_coordinates(Stheta)

    cylinder_frame.set_data(cylinder2world)
    cylinder.set_data(cylinder2world)
    prev_cylinder2world[:, :] = transform_from_exponential_coordinates(Stheta)
    return cylinder_frame, cylinder


fig = pv.figure()

mass = 1.0
length = 0.5
radius = 0.1
cylinder = fig.plot_cylinder(length=length, radius=radius)
cylinder2world = np.eye(4)
cylinder_frame = fig.plot_transform(A2B=cylinder2world, s=0.5)
inertia = spatial_inertia_of_cylinder(mass=mass, length=length, radius=radius)
inertia_inv = np.linalg.inv(inertia)
twist = np.zeros(6)

fig.plot_transform(A2B=np.eye(4), s=0.5)

fig.view_init()

n_frames = 10000
fig.animate(
    animation_callback, n_frames, fargs=(n_frames, cylinder, cylinder_frame, cylinder2world, inertia_inv, twist), loop=True)

fig.show()
