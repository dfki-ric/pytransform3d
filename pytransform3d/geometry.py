"""Basic functionality for geometrical shapes."""
import math
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
            [0, 2, 6],
            [0, 4, 5],
            [0, 1, 2],
            [1, 3, 2],
            [1, 5, 7],
            [1, 7, 3],
            [5, 1, 0],
            [5, 6, 7],
            [6, 2, 3],
            [6, 3, 7],
            [6, 4, 0],
            [6, 5, 4],
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

    def mesh(self, n_steps=20):
        vertices = np.empty((2 * n_steps * (n_steps - 1) + 2, 3))

        vertices[0] = np.array([0.0, 0.0, self.radius])
        vertices[1] = np.array([0.0, 0.0, -self.radius])
        step = math.pi / n_steps
        for i in range(1, n_steps):
            alpha = step * i
            base = 2 + 2 * n_steps * (i - 1)
            for j in range(2 * n_steps):
                theta = step * j
                vertices[base + j] = np.array([
                    math.sin(alpha) * math.cos(theta),
                    math.sin(alpha) * math.sin(theta),
                    math.cos(alpha)]) * self.radius

        triangles = []

        for j in range(2 * n_steps):
            j1 = (j + 1) % (2 * n_steps)
            base = 2
            triangles.append(np.array([0, base + j, base + j1]))
            base = 2 + 2 * n_steps * (n_steps - 2)
            triangles.append(np.array([1, base + j1, base + j]))

        for i in range(1, n_steps - 1):
            base1 = 2 + 2 * n_steps * (i - 1)
            base2 = base1 + 2 * n_steps
            for j in range(2 * n_steps):
                j1 = (j + 1) % (2 * n_steps)
                triangles.append(np.array([base2 + j, base1 + j1, base1 + j]))
                triangles.append(np.array([base2 + j, base2 + j1, base1 + j1]))

        return vertices, np.row_stack(triangles)


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

    def mesh(self, n_steps_circle=20, n_steps_length=4):
        vertices = np.empty((n_steps_circle * (n_steps_length + 1) + 2, 3))
        vertices[0] = np.array([0.0, 0.0, self.length * 0.5])
        vertices[1] = np.array([0.0, 0.0, -self.length * 0.5])
        step = math.pi * 2.0 / n_steps_circle
        h_step = self.length / n_steps_length
        for i in range(n_steps_length + 1):
            for j in range(n_steps_circle):
                theta = step * j
                vertices[2 + n_steps_circle * i + j] = np.array([
                    math.cos(theta) * self.radius,
                    math.sin(theta) * self.radius,
                    self.length * 0.5 - h_step * i])

        triangles = []
        for j in range(n_steps_circle):
            j1 = (j + 1) % n_steps_circle
            base = 2
            triangles.append(np.array([0, base + j, base + j1], dtype=int))
            base = 2 + n_steps_circle * n_steps_length
            triangles.append(np.array([1, base + j1, base + j], dtype=int))

        for i in range(n_steps_length):
            base1 = 2 + n_steps_circle * i
            base2 = base1 + n_steps_circle
            for j in range(n_steps_circle):
                j1 = (j + 1) % n_steps_circle
                triangles.append(np.array([base2 + j, base1 + j1, base1 + j]))
                triangles.append(np.array([base2 + j, base2 + j1, base1 + j1]))

        return vertices, np.row_stack(triangles)


class Mesh(GeometricShape):
    def __init__(self, pose, vertices, triangles):
        super(Mesh, self).__init__(pose)
        self.vertices = vertices
        self.triangles = triangles

    def mesh(self):
        return self.vertices, self.triangles


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
