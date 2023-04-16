"""Basic functionality for geometrical shapes."""
import numpy as np
from .transformations import transform, vectors_to_points


class GeometricShape(object):
    def __init__(self, pose):
        self.pose = pose


class Box(GeometricShape):
    def __init__(self, pose, size):
        super(Box, self).__init__(pose)
        self.size = size


class Sphere(GeometricShape):
    def __init__(self, pose, radius):
        super(Sphere, self).__init__(pose)
        self.radius = radius

    def surface(self, n_steps):
        phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j,
                              0.0:2.0 * np.pi:n_steps * 1j]
        x = self.pose[0, 3] + self.radius * np.sin(phi) * np.cos(theta)
        y = self.pose[1, 3] + self.radius * np.sin(phi) * np.sin(theta)
        z = self.pose[2, 3] + self.radius * np.cos(phi)
        return x, y, z


class Cylinder(GeometricShape):
    def __init__(self, pose, radius, length):
        super(Cylinder, self).__init__(pose)
        self.radius = radius
        self.length = length

    def surface(self, n_steps):
        phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = np.zeros_like(phi)
        z[len(z) // 2:] -= 0.5 * self.length
        z[:len(z) // 2] += 0.5 * self.length

        shape = x.shape

        P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
        P = transform(self.pose, vectors_to_points(P))[:, :3]

        x = P[:, 0].reshape(*shape)
        y = P[:, 1].reshape(*shape)
        z = P[:, 2].reshape(*shape)

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
        radius_x, radius_y, radius_z = self.radii

        phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j,
                              0.0:2.0 * np.pi:n_steps * 1j]
        x = radius_x * np.sin(phi) * np.cos(theta)
        y = radius_y * np.sin(phi) * np.sin(theta)
        z = radius_z * np.cos(phi)

        shape = x.shape

        P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
        P = transform(self.pose, vectors_to_points(P))[:, :3]

        x = P[:, 0].reshape(*shape)
        y = P[:, 1].reshape(*shape)
        z = P[:, 2].reshape(*shape)

        return x, y, z


class Capsule(GeometricShape):
    def __init__(self, pose, height, radius):
        super(Capsule, self).__init__(pose)
        self.height = height
        self.radius = radius

    def surface(self, n_steps):
        phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
        x = self.radius * np.sin(phi) * np.cos(theta)
        y = self.radius * np.sin(phi) * np.sin(theta)
        z = self.radius * np.cos(phi)
        z[len(z) // 2:] -= 0.5 * self.height
        z[:len(z) // 2] += 0.5 * self.height

        shape = x.shape

        P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
        P = transform(self.pose, vectors_to_points(P))[:, :3]

        x = P[:, 0].reshape(*shape)
        y = P[:, 1].reshape(*shape)
        z = P[:, 2].reshape(*shape)

        return x, y, z


class Cone(GeometricShape):
    def __init__(self, pose, height, radius):
        super(Cone, self).__init__(pose)
        self.height = height
        self.radius = radius
