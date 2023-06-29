"""Basic functionality for geometrical shapes."""
import numpy as np
from .transformations import transform, vectors_to_points


def unit_sphere_surface_grid(n_steps):
    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j,
                          0.0:2.0 * np.pi:n_steps * 1j]
    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def transform_surface(pose, x, y, z):
    shape = x.shape
    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    P = transform(pose, vectors_to_points(P))[:, :3]
    x = P[:, 0].reshape(*shape)
    y = P[:, 1].reshape(*shape)
    z = P[:, 2].reshape(*shape)
    return x, y, z
