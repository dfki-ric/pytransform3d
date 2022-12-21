"""Visualizer artists."""
import warnings
from itertools import chain
import numpy as np
import open3d as o3d
from .. import rotations as pr
from .. import transformations as pt
from .. import urdf


class Artist:
    """Abstract base class for objects that can be rendered."""

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        for g in self.geometries:
            figure.add_geometry(g)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return []


class Line3D(Artist):
    """A line.

    Parameters
    ----------
    P : array-like, shape (n_points, 3)
        Points of which the line consists.

    c : array-like, shape (n_points - 1, 3) or (3,), optional (default: black)
        Color can be given as individual colors per line segment or as one
        color for each segment. A color is represented by 3 values between
        0 and 1 indicate representing red, green, and blue respectively.
    """
    def __init__(self, P, c=(0, 0, 0)):
        self.line_set = o3d.geometry.LineSet()
        self.set_data(P, c)

    def set_data(self, P, c=None):
        """Update data.

        Parameters
        ----------
        P : array-like, shape (n_points, 3)
            Points of which the line consists.

        c : array-like, shape (n_points - 1, 3) or (3,), optional (default: black)
            Color can be given as individual colors per line segment or
            as one color for each segment. A color is represented by 3
            values between 0 and 1 indicate representing red, green, and
            blue respectively.
        """
        self.line_set.points = o3d.utility.Vector3dVector(P)
        self.line_set.lines = o3d.utility.Vector2iVector(np.hstack((
            np.arange(len(P) - 1)[:, np.newaxis],
            np.arange(1, len(P))[:, np.newaxis])))

        if c is not None:
            try:
                if len(c[0]) == 3:
                    self.line_set.colors = o3d.utility.Vector3dVector(c)
            except TypeError:  # one color for all segments
                self.line_set.colors = o3d.utility.Vector3dVector(
                    [c for _ in range(len(P) - 1)])

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.line_set]


class PointCollection3D(Artist):
    """Collection of points.

    Parameters
    ----------
    P : array, shape (n_points, 3)
        Points

    s : float, optional (default: 0.05)
        Scaling of the spheres that will be drawn.

    c : array-like, shape (3,) or (n_points, 3), optional (default: black)
        A color is represented by 3 values between 0 and 1 indicate
        representing red, green, and blue respectively.
    """
    def __init__(self, P, s=0.05, c=None):
        self._points = []
        self._P = np.zeros_like(P)

        if c is not None:
            c = np.asarray(c)
            if c.ndim == 1:
                c = np.tile(c, (len(P), 1))

        for i in range(len(P)):
            point = o3d.geometry.TriangleMesh.create_sphere(
                radius=s, resolution=10)
            if c is not None:
                n_vertices = len(point.vertices)
                colors = np.zeros((n_vertices, 3))
                colors[:] = c[i]
                point.vertex_colors = o3d.utility.Vector3dVector(colors)
            point.compute_vertex_normals()
            self._points.append(point)

        self.set_data(P)

    def set_data(self, P):
        """Update data.

        Parameters
        ----------
        P : array, shape (n_points, 3)
            Points
        """
        P = np.copy(P)
        for i, point, p, previous_p in zip(
                range(len(self._P)), self._points, P, self._P):
            if any(np.isnan(p)):
                P[i] = previous_p
            else:
                point.translate(p - previous_p)

        self._P = P

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return self._points


class Vector3D(Artist):
    """A vector.

    Parameters
    ----------
    start : array-like, shape (3,), optional (default: [0, 0, 0])
        Start of the vector

    direction : array-like, shape (3,), optional (default: [1, 0, 0])
        Direction of the vector

    c : array-like, shape (3,), optional (default: black)
        A color is represented by 3 values between 0 and 1 indicate
        representing red, green, and blue respectively.
    """
    def __init__(self, start=np.zeros(3), direction=np.array([1, 0, 0]),
                 c=(0, 0, 0)):
        self.vector = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.035, cone_radius=0.07,
            cylinder_height=0.8, cone_height=0.2)
        self.set_data(start, direction, c)

    def set_data(self, start, direction, c=None):
        """Update data.

        Parameters
        ----------
        start : array-like, shape (3,)
            Start of the vector

        direction : array-like, shape (3,)
            Direction of the vector

        c : array-like, shape (3,), optional (default: black)
            A color is represented by 3 values between 0 and 1 indicate
            representing red, green, and blue respectively.
        """
        length = np.linalg.norm(direction)
        z = direction / length
        x, y = pr.plane_basis_from_normal(z)
        R = np.column_stack((x, y, z))
        A2B = pt.transform_from(R, start)

        new_vector = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.035 * length, cone_radius=0.07 * length,
            cylinder_height=0.8 * length, cone_height=0.2 * length)
        self.vector.vertices = new_vector.vertices
        self.vector.triangles = new_vector.triangles
        self.vector.transform(A2B)
        if c is not None:
            self.vector.paint_uniform_color(c)
        self.vector.compute_vertex_normals()

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.vector]


class Frame(Artist):
    """Coordinate frame.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors
    """
    def __init__(self, A2B, label=None, s=1.0):
        self.A2B = None
        self.label = None
        self.s = s

        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.s)

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B
        self.label = label
        if label is not None:
            warnings.warn(
                "This viewer does not support text. Frame label "
                "will be ignored.")

        self.frame.transform(
            pt.invert_transform(previous_A2B, check=False))
        self.frame.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.frame]


class Trajectory(Artist):
    """Trajectory of poses.

    Parameters
    ----------
    H : array-like, shape (n_steps, 4, 4)
        Sequence of poses represented by homogeneous matrices

    n_frames : int, optional (default: 10)
        Number of frames that should be plotted to indicate the rotation

    s : float, optional (default: 1)
        Scaling of the frames that will be drawn

    c : array-like, shape (3,), optional (default: black)
        A color is represented by 3 values between 0 and 1 indicate
        representing red, green, and blue respectively.
    """
    def __init__(self, H, n_frames=10, s=1.0, c=(0, 0, 0)):
        self.H = H
        self.n_frames = n_frames
        self.s = s
        self.c = c

        self.key_frames = []
        self.line = Line3D(H[:, :3, 3], c)

        self.key_frames_indices = np.linspace(
            0, len(self.H) - 1, self.n_frames, dtype=np.int64)
        for key_frame_idx in self.key_frames_indices:
            self.key_frames.append(Frame(self.H[key_frame_idx], s=self.s))

        self.set_data(H)

    def set_data(self, H):
        """Update data.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Sequence of poses represented by homogeneous matrices
        """
        self.line.set_data(H[:, :3, 3])
        for i, key_frame_idx in enumerate(self.key_frames_indices):
            self.key_frames[i].set_data(H[key_frame_idx])

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return self.line.geometries + list(
            chain(*[kf.geometries for kf in self.key_frames]))


class Sphere(Artist):
    """Sphere.

    Parameters
    ----------
    radius : float, optional (default: 1)
        Radius of the sphere

    A2B : array-like, shape (4, 4)
        Center of the sphere

    resolution : int, optional (default: 20)
        The resolution of the sphere. The longitudes will be split into
        resolution segments (i.e. there are resolution + 1 latitude lines
        including the north and south pole). The latitudes will be split
        into 2 * resolution segments (i.e. there are 2 * resolution
        longitude lines.)

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, radius=1.0, A2B=np.eye(4), resolution=20, c=None):
        self.sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius, resolution)
        if c is not None:
            n_vertices = len(self.sphere.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.sphere.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.sphere.compute_vertex_normals()
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the sphere.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.sphere.transform(
            pt.invert_transform(previous_A2B, check=False))
        self.sphere.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.sphere]


class Box(Artist):
    """Box.

    Parameters
    ----------
    size : array-like, shape (3,), optional (default: [1, 1, 1])
        Size of the box per dimension

    A2B : array-like, shape (4, 4), optional (default: I)
        Center of the box

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, size=np.ones(3), A2B=np.eye(4), c=None):
        self.half_size = np.asarray(size) / 2.0
        width, height, depth = size
        self.box = o3d.geometry.TriangleMesh.create_box(
            width, height, depth)
        if c is not None:
            n_vertices = len(self.box.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.box.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.box.compute_vertex_normals()
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the box.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            self.box.transform(
                pt.transform_from(R=np.eye(3), p=-self.half_size))
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.box.transform(pt.invert_transform(previous_A2B, check=False))
        self.box.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.box]


class Cylinder(Artist):
    """Cylinder.

    A cylinder is the volume covered by a disk moving along a line segment.

    Parameters
    ----------
    length : float, optional (default: 1)
        Length of the cylinder.

    radius : float, optional (default: 1)
        Radius of the cylinder.

    A2B : array-like, shape (4, 4)
        Pose of the cylinder. The position corresponds to the center of the
        line segment and the z-axis to the direction of the line segment.

    resolution : int, optional (default: 20)
        The circles will be split into resolution segments.

    split : int, optional (default: 4)
        The height will be split into split segments

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, length=2.0, radius=1.0, A2B=np.eye(4), resolution=20,
                 split=4, c=None):
        self.cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=length, resolution=resolution,
            split=split)
        if c is not None:
            n_vertices = len(self.cylinder.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.cylinder.vertex_colors = \
                o3d.utility.Vector3dVector(colors)
        self.cylinder.compute_vertex_normals()
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the cylinder.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.cylinder.transform(
            pt.invert_transform(previous_A2B, check=False))
        self.cylinder.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.cylinder]


class Mesh(Artist):
    """Mesh.

    Parameters
    ----------
    filename : str
        Path to mesh file

    A2B : array-like, shape (4, 4)
        Center of the mesh

    s : array-like, shape (3,), optional (default: [1, 1, 1])
        Scaling of the mesh that will be drawn

    c : array-like, shape (n_vertices, 3) or (3,), optional (default: None)
        Color(s)
    """
    def __init__(self, filename, A2B=np.eye(4), s=np.ones(3), c=None):
        self.mesh = o3d.io.read_triangle_mesh(filename)
        self.mesh.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.mesh.vertices) * s)
        self.mesh.compute_vertex_normals()
        if c is not None:
            n_vertices = len(self.mesh.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the mesh.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.mesh.transform(pt.invert_transform(previous_A2B, check=False))
        self.mesh.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]


class Ellipsoid(Artist):
    """Ellipsoid.

    Parameters
    ----------
    radii : array-like, shape (3,)
        Radii along the x-axis, y-axis, and z-axis of the ellipsoid.

    A2B : array-like, shape (4, 4)
        Pose of the ellipsoid.

    resolution : int, optional (default: 20)
        The resolution of the ellipsoid. The longitudes will be split into
        resolution segments (i.e. there are resolution + 1 latitude lines
        including the north and south pole). The latitudes will be split
        into 2 * resolution segments (i.e. there are 2 * resolution
        longitude lines.)

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, radii, A2B=np.eye(4), resolution=20, c=None):
        self.ellipsoid = o3d.geometry.TriangleMesh.create_sphere(
            1.0, resolution)
        vertices = np.asarray(self.ellipsoid.vertices)
        vertices *= np.asarray(radii)[np.newaxis]
        self.ellipsoid.vertices = o3d.utility.Vector3dVector(vertices)
        self.ellipsoid.compute_vertex_normals()
        if c is not None:
            n_vertices = len(self.ellipsoid.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.ellipsoid.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the ellipsoid.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.ellipsoid.transform(
            pt.invert_transform(previous_A2B, check=False))
        self.ellipsoid.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.ellipsoid]


class Capsule(Artist):
    """Capsule.

    A capsule is the volume covered by a sphere moving along a line segment.

    Parameters
    ----------
    height : float, optional (default: 1)
        Height of the capsule along its z-axis.

    radius : float, optional (default: 1)
        Radius of the capsule.

    A2B : array-like, shape (4, 4)
        Pose of the capsule. The position corresponds to the center of the line
        segment and the z-axis to the direction of the line segment.

    resolution : int, optional (default: 20)
        The resolution of the half spheres. The longitudes will be split into
        resolution segments (i.e. there are resolution + 1 latitude lines
        including the north and south pole). The latitudes will be split
        into 2 * resolution segments (i.e. there are 2 * resolution
        longitude lines.)

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, height=1, radius=1, A2B=np.eye(4), resolution=20,
                 c=None):
        self.capsule = o3d.geometry.TriangleMesh.create_sphere(
            radius, resolution)
        vertices = np.asarray(self.capsule.vertices)
        vertices[vertices[:, 2] < 0, 2] -= 0.5 * height
        vertices[vertices[:, 2] > 0, 2] += 0.5 * height
        self.capsule.vertices = o3d.utility.Vector3dVector(vertices)
        self.capsule.compute_vertex_normals()
        if c is not None:
            n_vertices = len(self.capsule.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.capsule.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Start of the capsule's line segment.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.capsule.transform(
            pt.invert_transform(previous_A2B, check=False))
        self.capsule.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.capsule]


class Cone(Artist):
    """Cone.

    Parameters
    ----------
    height : float, optional (default: 1)
        Height of the cone along its z-axis.

    radius : float, optional (default: 1)
        Radius of the cone.

    A2B : array-like, shape (4, 4)
        Pose of the cone, which is the center of its circle.

    resolution : int, optional (default: 20)
        The circle will be split into resolution segments.

    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, height=1, radius=1, A2B=np.eye(4), resolution=20,
                 c=None):
        self.cone = o3d.geometry.TriangleMesh.create_cone(
            radius, height, resolution)
        self.cone.compute_vertex_normals()
        if c is not None:
            n_vertices = len(self.cone.vertices)
            colors = np.zeros((n_vertices, 3))
            colors[:] = c
            self.cone.vertex_colors = o3d.utility.Vector3dVector(colors)
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the cone's circle.
        """
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.cone.transform(
            pt.invert_transform(previous_A2B, check=False))
        self.cone.transform(self.A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.cone]


class Plane(Artist):
    """Plane.

    The plane will be defined either by a normal and a point in the plane or
    by the Hesse normal form, which only needs a normal and the distance to
    the origin from which we can compute the point in the plane as d * normal.

    A plane will be visualized by a square.

    Parameters
    ----------
    normal : array-like, shape (3,), optional (default: [0, 0, 1])
        Plane normal.

    d : float, optional (default: None)
        Distance to origin in Hesse normal form.

    point_in_plane : array-like, shape (3,), optional (default: None)
        Point in plane.

    s : float, optional (default: 1)
        Scaling of the plane that will be drawn.

    c : array-like, shape (3,), optional (default: None)
        Color.
    """
    def __init__(self, normal=np.array([0.0, 0.0, 1.0]), d=None,
                 point_in_plane=None, s=1.0, c=None):
        self.triangles = np.array([
            [0, 1, 2],
            [1, 3, 2],
            [2, 1, 0],
            [2, 3, 1],
        ])
        vertices = np.zeros((4, 3))
        self.plane = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(self.triangles))

        self.set_data(normal, d, point_in_plane, s, c)

    def set_data(self, normal, d=None, point_in_plane=None, s=None,
                 c=None):
        """Update data.

        Parameters
        ----------
        normal : array-like, shape (3,)
            Plane normal.

        d : float, optional (default: None)
            Distance to origin in Hesse normal form.

        point_in_plane : array-like, shape (3,), optional (default: None)
            Point in plane.

        s : float, optional (default: 1)
            Scaling of the plane that will be drawn.

        c : array-like, shape (3,), optional (default: None)
            Color.

        Raises
        ------
        ValueError
            If neither 'd' nor 'point_in_plane' is defined.
        """
        normal = np.asarray(normal)
        if point_in_plane is None:
            if d is None:
                raise ValueError(
                    "Either 'd' or 'point_in_plane' has to be defined!")
            point_in_plane = d * normal
        else:
            point_in_plane = np.asarray(point_in_plane)

        x_axis, y_axis = pr.plane_basis_from_normal(normal)
        vertices = np.array([
            point_in_plane + s * x_axis + s * y_axis,
            point_in_plane - s * x_axis + s * y_axis,
            point_in_plane + s * x_axis - s * y_axis,
            point_in_plane - s * x_axis - s * y_axis,
        ])
        self.plane.vertices = o3d.utility.Vector3dVector(vertices)
        self.plane.compute_vertex_normals()
        if c is not None:
            self.plane.paint_uniform_color(c)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.plane]


class Camera(Artist):
    """Camera.

    Parameters
    ----------
    M : array-like, shape (3, 3)
        Intrinsic camera matrix that contains the focal lengths on the diagonal
        and the center of the the image in the last column. It does not matter
        whether values are given in meters or pixels as long as the unit is the
        same as for the sensor size.

    cam2world : array-like, shape (4, 4), optional (default: I)
        Transformation matrix of camera in world frame. We assume that the
        position is given in meters.

    virtual_image_distance : float, optional (default: 1)
        Distance from pinhole to virtual image plane that will be displayed.
        We assume that this distance is given in meters. The unit has to be
        consistent with the unit of the position in cam2world.

    sensor_size : array-like, shape (2,), optional (default: [1920, 1080])
        Size of the image sensor: (width, height). It does not matter whether
        values are given in meters or pixels as long as the unit is the same as
        for the sensor size.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.
    """
    def __init__(self, M, cam2world=None, virtual_image_distance=1,
                 sensor_size=(1920, 1080), strict_check=True):
        self.M = None
        self.cam2world = None
        self.virtual_image_distance = None
        self.sensor_size = None
        self.strict_check = strict_check

        self.line_set = o3d.geometry.LineSet()

        if cam2world is None:
            cam2world = np.eye(4)

        self.set_data(M, cam2world, virtual_image_distance, sensor_size)

    def set_data(self, M=None, cam2world=None, virtual_image_distance=None,
                 sensor_size=None):
        """Update camera parameters.

        Parameters
        ----------
        M : array-like, shape (3, 3), optional (default: old value)
            Intrinsic camera matrix that contains the focal lengths on the
            diagonal and the center of the the image in the last column. It
            does not matter whether values are given in meters or pixels as
            long as the unit is the same as for the sensor size.

        cam2world : array-like, shape (4, 4), optional (default: old value)
            Transformation matrix of camera in world frame. We assume that
            the position is given in meters.

        virtual_image_distance : float, optional (default: old value)
            Distance from pinhole to virtual image plane that will be
            displayed. We assume that this distance is given in meters.
            The unit has to be consistent with the unit of the position
            in cam2world.

        sensor_size : array-like, shape (2,), optional (default: old value)
            Size of the image sensor: (width, height). It does not matter
            whether values are given in meters or pixels as long as the
            unit is the same as for the sensor size.
        """
        if M is not None:
            self.M = M
        if cam2world is not None:
            self.cam2world = pt.check_transform(
                cam2world, strict_check=self.strict_check)
        if virtual_image_distance is not None:
            self.virtual_image_distance = virtual_image_distance
        if sensor_size is not None:
            self.sensor_size = sensor_size

        camera_center_in_world = cam2world[:3, 3]
        focal_length = np.mean(np.diag(M[:2, :2]))
        sensor_corners_in_cam = np.array([
            [0, 0, focal_length],
            [0, sensor_size[1], focal_length],
            [sensor_size[0], sensor_size[1], focal_length],
            [sensor_size[0], 0, focal_length],
        ])
        sensor_corners_in_cam[:, 0] -= M[0, 2]
        sensor_corners_in_cam[:, 1] -= M[1, 2]
        sensor_corners_in_world = pt.transform(
            cam2world, pt.vectors_to_points(sensor_corners_in_cam))[:, :3]
        virtual_image_corners = (
            virtual_image_distance / focal_length *
            (sensor_corners_in_world - camera_center_in_world[np.newaxis]) +
            camera_center_in_world[np.newaxis])

        up = virtual_image_corners[0] - virtual_image_corners[1]
        camera_line_points = np.vstack((
            camera_center_in_world[:3],
            virtual_image_corners[0],
            virtual_image_corners[1],
            virtual_image_corners[2],
            virtual_image_corners[3],
            virtual_image_corners[0] + 0.1 * up,
            0.5 * (virtual_image_corners[0] +
                   virtual_image_corners[3]) + 0.5 * up,
            virtual_image_corners[3] + 0.1 * up
        ))

        self.line_set.points = o3d.utility.Vector3dVector(
            camera_line_points)
        self.line_set.lines = o3d.utility.Vector2iVector(
            np.array([[0, 1], [0, 2], [0, 3], [0, 4],
                      [1, 2], [2, 3], [3, 4], [4, 1],
                      [5, 6], [6, 7], [7, 5]]))

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.line_set]


class Graph(Artist):
    """Graph of connected frames.

    Parameters
    ----------
    tm : TransformManager
        Representation of the graph

    frame : str
        Name of the base frame in which the graph will be displayed

    show_frames : bool, optional (default: False)
        Show coordinate frames

    show_connections : bool, optional (default: False)
        Draw lines between frames of the graph

    show_visuals : bool, optional (default: False)
        Show visuals that are stored in the graph

    show_collision_objects : bool, optional (default: False)
        Show collision objects that are stored in the graph

    show_name : bool, optional (default: False)
        Show names of frames

    whitelist : list, optional (default: all)
        List of frames that should be displayed

    s : float, optional (default: 1)
        Scaling of the frames that will be drawn
    """
    def __init__(self, tm, frame, show_frames=False, show_connections=False,
                 show_visuals=False, show_collision_objects=False,
                 show_name=False, whitelist=None, s=1.0):
        self.tm = tm
        self.frame = frame
        self.show_frames = show_frames
        self.show_connections = show_connections
        self.show_visuals = show_visuals
        self.show_collision_objects = show_collision_objects
        self.whitelist = whitelist
        self.s = s

        if self.frame not in self.tm.nodes:
            raise KeyError("Unknown frame '%s'" % self.frame)

        self.nodes = list(sorted(self.tm._whitelisted_nodes(whitelist)))

        self.frames = {}
        if self.show_frames:
            for node in self.nodes:
                try:
                    node2frame = self.tm.get_transform(node, frame)
                    name = node if show_name else None
                    self.frames[node] = Frame(node2frame, name, self.s)
                except KeyError:
                    pass  # Frame is not connected to the reference frame

        self.connections = {}
        if self.show_connections:
            for frame_names in self.tm.transforms.keys():
                from_frame, to_frame = frame_names
                if (from_frame in self.tm.nodes and
                        to_frame in self.tm.nodes):
                    try:
                        self.tm.get_transform(from_frame, self.frame)
                        self.tm.get_transform(to_frame, self.frame)
                        self.connections[frame_names] = \
                            o3d.geometry.LineSet()
                    except KeyError:
                        pass  # Frame is not connected to reference frame

        self.visuals = {}
        if show_visuals and hasattr(self.tm, "visuals"):
            self.visuals.update(_objects_to_artists(self.tm.visuals))
        self.collision_objects = {}
        if show_collision_objects and hasattr(
                self.tm, "collision_objects"):
            self.collision_objects.update(
                _objects_to_artists(self.tm.collision_objects))

        self.set_data()

    def set_data(self):
        """Indicate that data has been updated."""
        if self.show_frames:
            for node in self.nodes:
                try:
                    node2frame = self.tm.get_transform(node, self.frame)
                    self.frames[node].set_data(node2frame)
                except KeyError:
                    pass  # Frame is not connected to the reference frame

        if self.show_connections:
            for frame_names in self.connections:
                from_frame, to_frame = frame_names
                try:
                    from2ref = self.tm.get_transform(
                        from_frame, self.frame)
                    to2ref = self.tm.get_transform(to_frame, self.frame)

                    points = np.vstack((from2ref[:3, 3], to2ref[:3, 3]))
                    self.connections[frame_names].points = \
                        o3d.utility.Vector3dVector(points)
                    self.connections[frame_names].lines = \
                        o3d.utility.Vector2iVector(np.array([[0, 1]]))
                except KeyError:
                    pass  # Frame is not connected to the reference frame

        for frame, obj in self.visuals.items():
            A2B = self.tm.get_transform(frame, self.frame)
            obj.set_data(A2B)

        for frame, obj in self.collision_objects.items():
            A2B = self.tm.get_transform(frame, self.frame)
            obj.set_data(A2B)

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        geometries = []
        if self.show_frames:
            for f in self.frames.values():
                geometries += f.geometries
        if self.show_connections:
            geometries += list(self.connections.values())
        for obj in self.visuals.values():
            geometries += obj.geometries
        for obj in self.collision_objects.values():
            geometries += obj.geometries
        return geometries


def _objects_to_artists(objects):
    """Convert geometries from URDF to artists.

    Parameters
    ----------
    objects : list of Geometry
        Objects parsed from URDF.

    Returns
    -------
    artists : dict
        Mapping from frame names to artists.
    """
    artists = {}
    for obj in objects:
        if obj.color is None:
            color = None
        else:
            # we loose the alpha channel as it is not supported by Open3D
            color = (obj.color[0], obj.color[1], obj.color[2])
        try:
            if isinstance(obj, urdf.Sphere):
                artist = Sphere(radius=obj.radius, c=color)
            elif isinstance(obj, urdf.Box):
                artist = Box(obj.size, c=color)
            elif isinstance(obj, urdf.Cylinder):
                artist = Cylinder(obj.length, obj.radius, c=color)
            else:
                assert isinstance(obj, urdf.Mesh)
                artist = Mesh(obj.filename, s=obj.scale, c=color)
            artists[obj.frame] = artist
        except RuntimeError as e:
            warnings.warn(str(e))
    return artists
