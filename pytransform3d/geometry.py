"""Basic functionality for geometrical shapes."""
from itertools import product
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


def transform_vertices(pose, vertices):
    return transform(pose, vectors_to_points(vertices))[:, :3]


BOX_COORDS = np.array(list(product([-0.5, 0.5], repeat=3)))


class GeometricShape(object):
    def __init__(self, pose):
        self.pose = pose


class Box(GeometricShape):
    def __init__(self, pose, size):
        super(Box, self).__init__(pose)
        self.size = size

    def surface(self, n_steps):
        x, y, z = unit_sphere_surface_grid(n_steps)

        max_distance_to_center = np.linalg.norm(self.size)

        x *= max_distance_to_center
        y *= max_distance_to_center
        z *= max_distance_to_center

        x = np.clip(x, -0.5 * self.size[0], 0.5 * self.size[0])
        y = np.clip(y, -0.5 * self.size[1], 0.5 * self.size[1])
        z = np.clip(z, -0.5 * self.size[2], 0.5 * self.size[2])

        x, y, z = transform_surface(self.pose, x, y, z)

        return x, y, z

    def mesh(self):
        vertices = BOX_COORDS * self.size
        vertices = transform_vertices(self.pose, vertices)
        triangles = np.array([
            [0, 1, 2],
            [2, 1, 3],
            [0, 4, 5],
            [1, 0, 5],
            [2, 7, 6],
            [2, 3, 7],
            [0, 6, 4],
            [0, 2, 6],
            [3, 1, 5],
            [3, 5, 7],
            [5, 4, 6],
            [5, 6, 7],
        ], dtype=int)
        return vertices, triangles


class Sphere(GeometricShape):
    def __init__(self, pose, radius):
        super(Sphere, self).__init__(pose)
        self.radius = radius

    def surface(self, n_steps):
        x, y, z = unit_sphere_surface_grid(n_steps)

        x *= self.radius
        y *= self.radius
        z *= self.radius

        x, y, z = transform_surface(self.pose, x, y, z)

        return x, y, z


class Cylinder(GeometricShape):
    def __init__(self, pose, radius, length):
        super(Cylinder, self).__init__(pose)
        self.radius = radius
        self.length = length

    def surface(self, n_steps):
        x, y, z = unit_sphere_surface_grid(n_steps)

        x *= self.radius
        y *= self.radius
        z[len(z) // 2:] = -0.5 * self.length
        z[:len(z) // 2] = 0.5 * self.length

        x, y, z = transform_surface(self.pose, x, y, z)

        return x, y, z


class Mesh(GeometricShape):
    def __init__(self, pose, vertices, triangles):
        super(Mesh, self).__init__(pose)
        self.vertices = vertices
        self.triangles = triangles


class Ellipsoid(GeometricShape):
    def __init__(self, pose, radii):
        super(Ellipsoid, self).__init__(pose)
        self.radii = radii

    def surface(self, n_steps):
        x, y, z = unit_sphere_surface_grid(n_steps)

        x *= self.radii[0]
        y *= self.radii[1]
        z *= self.radii[2]

        x, y, z = transform_surface(self.pose, x, y, z)

        return x, y, z


class Capsule(GeometricShape):
    def __init__(self, pose, height, radius):
        super(Capsule, self).__init__(pose)
        self.height = height
        self.radius = radius

    def surface(self, n_steps):
        x, y, z = unit_sphere_surface_grid(n_steps)

        x *= self.radius
        y *= self.radius
        z *= self.radius
        z[len(z) // 2:] -= 0.5 * self.height
        z[:len(z) // 2] += 0.5 * self.height

        x, y, z = transform_surface(self.pose, x, y, z)

        return x, y, z


class Cone(GeometricShape):
    def __init__(self, pose, height, radius):
        super(Cone, self).__init__(pose)
        self.height = height
        self.radius = radius

    def surface(self, n_steps):
        x, y, z = unit_sphere_surface_grid(n_steps)
        x[len(x) // 2:] = 0.0
        y[len(y) // 2:] = 0.0
        z[:len(z) // 2, :] = 0.0

        z[len(z) // 2:] = self.height
        x *= self.radius
        y *= self.radius

        x, y, z = transform_surface(self.pose, x, y, z)

        return x, y, z