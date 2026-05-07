"""Visualizer artists for the viser backend."""

import warnings
from itertools import chain

import numpy as np
import trimesh
import trimesh.creation
import trimesh.util
import trimesh.visual

from .. import rotations as pr
from .. import transformations as pt
from .. import urdf


def _to_viser_color(c):
    """Convert color from [0, 1] floats to (r, g, b) tuple with 0-255 ints.

    Parameters
    ----------
    c : array-like, shape (3,) or None
        RGB color with values in [0, 1], or None.

    Returns
    -------
    color : tuple of int or None
        RGB color with values in [0, 255], or None.
    """
    if c is None:
        return None
    arr = np.asarray(c, dtype=float)
    return tuple(int(np.clip(v * 255.0, 0, 255)) for v in arr[:3])


def _wxyz_from_matrix(A2B):
    """Extract quaternion (wxyz) from a 4x4 homogeneous transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    wxyz : array, shape (4,)
        Quaternion in wxyz convention.
    """
    return pr.quaternion_from_matrix(A2B[:3, :3], strict_check=False)


def _make_arrow_mesh(length):
    """Create a trimesh arrow pointing along the z-axis.

    Parameters
    ----------
    length : float
        Total length of the arrow.

    Returns
    -------
    mesh : trimesh.Trimesh
        Combined arrow mesh (shaft + tip).
    """
    shaft_length = length * 0.8
    tip_length = length * 0.2
    shaft_radius = 0.035 * length
    tip_radius = 0.07 * length

    shaft = trimesh.creation.cylinder(
        radius=shaft_radius, height=shaft_length, sections=20
    )
    shaft.apply_translation([0.0, 0.0, shaft_length / 2.0])

    tip = trimesh.creation.cone(
        radius=tip_radius, height=tip_length, sections=20
    )
    tip.apply_translation([0.0, 0.0, shaft_length])

    return trimesh.util.concatenate([shaft, tip])


def _set_trimesh_color(mesh, c):
    """Apply a uniform color to a trimesh mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to colorize.

    c : array-like, shape (3,) or None
        RGB color with values in [0, 1], or None.
    """
    if c is None:
        return
    viser_color = _to_viser_color(c)
    n = len(mesh.vertices)
    rgba = np.array([list(viser_color) + [255]], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh,
        vertex_colors=np.tile(rgba, (n, 1)),
    )


class Artist:
    """Abstract base class for objects that can be rendered."""

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """

    def remove(self):
        """Remove artist from figure."""

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            Empty list (viser artists manage their own handles).
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
        0 and 1 indicating red, green, and blue respectively.
    """

    def __init__(self, P, c=(0, 0, 0)):
        self.P = np.asarray(P, dtype=float)
        self.c = c
        self._handle = None

    def _make_segments(self):
        return np.stack([self.P[:-1], self.P[1:]], axis=1).astype(np.float32)

    def _make_colors(self, n_segments):
        # viser add_line_segments broadcasts colors to points shape (N, 2, 3)
        # so colors must be (N, 2, 3), (N, 1, 3), (1, 1, 3) or similar
        c = np.asarray(self.c, dtype=float)
        if c.ndim == 2 and len(c) == n_segments:
            # Per-segment colors: shape (N, 3) -> (N, 1, 3) to broadcast
            rgb = (np.clip(c, 0.0, 1.0) * 255).astype(np.uint8)
            return rgb[:, np.newaxis, :]
        color_1d = c if c.ndim == 1 else c[0]
        rgb = _to_viser_color(color_1d)
        return np.array([[list(rgb)]], dtype=np.uint8)  # shape (1, 1, 3)

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        if len(self.P) < 2:
            return
        segments = self._make_segments()
        colors = self._make_colors(len(segments))
        self._handle = figure.scene.add_line_segments(
            name=figure._next_name("line"),
            points=segments,
            colors=colors,
        )

    def set_data(self, P, c=None):
        """Update data.

        Parameters
        ----------
        P : array-like, shape (n_points, 3)
            Points of which the line consists.

        c : array-like, shape (n_points - 1, 3) or (3,), optional
            (default: None). Color can be given as individual colors per line
            segment or as one color for each segment. A color is represented
            by 3 values between 0 and 1 indicating red, green, and blue.
        """
        self.P = np.asarray(P, dtype=float)
        if c is not None:
            self.c = c
        if self._handle is None or len(self.P) < 2:
            return
        segments = self._make_segments()
        colors = self._make_colors(len(segments))
        self._handle.points = segments
        self._handle.colors = colors

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class PointCollection3D(Artist):
    """Collection of points.

    Parameters
    ----------
    P : array, shape (n_points, 3)
        Points

    s : float, optional (default: 0.05)
        Scaling of the points that will be drawn.

    c : array-like, shape (3,) or (n_points, 3), optional (default: None)
        A color is represented by 3 values between 0 and 1 indicating
        red, green, and blue respectively.
    """

    def __init__(self, P, s=0.05, c=None):
        self.P = np.asarray(P, dtype=float)
        self.s = s
        self.c = c
        self._handle = None

    def _make_colors(self):
        n = len(self.P)
        if self.c is None:
            return np.zeros((n, 3), dtype=np.uint8)
        c = np.asarray(self.c, dtype=float)
        if c.ndim == 1:
            rgb = _to_viser_color(c)
            return np.tile(np.array([list(rgb)], dtype=np.uint8), (n, 1))
        return (np.clip(c, 0.0, 1.0) * 255).astype(np.uint8)

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        self._handle = figure.scene.add_point_cloud(
            name=figure._next_name("points"),
            points=self.P.astype(np.float32),
            colors=self._make_colors(),
            point_size=self.s,
        )

    def set_data(self, P):
        """Update data.

        Parameters
        ----------
        P : array, shape (n_points, 3)
            Points
        """
        self.P = np.asarray(P, dtype=float)
        if self._handle is None:
            return
        self._handle.points = self.P.astype(np.float32)
        self._handle.colors = self._make_colors()

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class Vector3D(Artist):
    """A vector.

    Parameters
    ----------
    start : array-like, shape (3,), optional (default: [0, 0, 0])
        Start of the vector

    direction : array-like, shape (3,), optional (default: [1, 0, 0])
        Direction of the vector

    c : array-like, shape (3,), optional (default: black)
        A color is represented by 3 values between 0 and 1 indicating
        red, green, and blue respectively.
    """

    def __init__(
        self, start=np.zeros(3), direction=np.array([1, 0, 0]), c=(0, 0, 0)
    ):
        self.start = np.asarray(start, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        self.c = c
        self._handle = None
        self._name = None
        self._figure = None

    def _pose_and_length(self):
        length = np.linalg.norm(self.direction)
        if length < 1e-10:
            return length, np.eye(4)
        z = self.direction / length
        x, y = pr.plane_basis_from_normal(z)
        R = np.column_stack((x, y, z))
        return length, pt.transform_from(R, self.start)

    def _add_to_scene(self):
        length, A2B = self._pose_and_length()
        if length < 1e-10:
            return None
        mesh = _make_arrow_mesh(length)
        _set_trimesh_color(mesh, self.c)
        return self._figure.scene.add_mesh_trimesh(
            name=self._name,
            mesh=mesh,
            wxyz=_wxyz_from_matrix(A2B),
            position=A2B[:3, 3],
        )

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        self._figure = figure
        self._name = figure._next_name("vector")
        self._handle = self._add_to_scene()

    def set_data(self, start, direction, c=None):
        """Update data.

        Parameters
        ----------
        start : array-like, shape (3,)
            Start of the vector

        direction : array-like, shape (3,)
            Direction of the vector

        c : array-like, shape (3,), optional (default: None)
            A color is represented by 3 values between 0 and 1 indicating
            red, green, and blue respectively.
        """
        self.start = np.asarray(start, dtype=float)
        self.direction = np.asarray(direction, dtype=float)
        if c is not None:
            self.c = c
        if self._figure is None:
            return
        if self._handle is not None:
            self._handle.remove()
        self._handle = self._add_to_scene()

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
        self.A2B = np.asarray(A2B, dtype=float)
        self.label = label
        self.s = s
        self._handle = None
        self._label_handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        name = figure._next_name("frame")
        self._handle = figure.scene.add_frame(
            name=name,
            axes_length=self.s,
            axes_radius=self.s / 20.0,
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )
        if self.label is not None:
            self._label_handle = figure.scene.add_label(
                name=name + "/label",
                text=self.label,
                position=self.A2B[:3, 3],
            )

    def set_data(self, A2B, label=None):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if label is not None:
            self.label = label
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]
        if self._label_handle is not None:
            self._label_handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        if self._label_handle is not None:
            self._label_handle.remove()
            self._label_handle = None


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
        A color is represented by 3 values between 0 and 1 indicating
        red, green, and blue respectively.
    """

    def __init__(self, H, n_frames=10, s=1.0, c=(0, 0, 0)):
        self.H = np.asarray(H, dtype=float)
        self.n_frames = n_frames
        self.s = s
        self.c = c

        self.key_frames = []
        self.line = Line3D(self.H[:, :3, 3], c)

        self.key_frames_indices = np.linspace(
            0, len(self.H) - 1, self.n_frames, dtype=np.int64
        )
        for key_frame_idx in self.key_frames_indices:
            self.key_frames.append(Frame(self.H[key_frame_idx], s=self.s))

        self.set_data(H)

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        self.line.add_artist(figure)
        for kf in self.key_frames:
            kf.add_artist(figure)

    def set_data(self, H):
        """Update data.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Sequence of poses represented by homogeneous matrices
        """
        self.H = np.asarray(H, dtype=float)
        self.line.set_data(self.H[:, :3, 3])
        for i, key_frame_idx in enumerate(self.key_frames_indices):
            self.key_frames[i].set_data(self.H[key_frame_idx])

    def remove(self):
        """Remove artist from figure."""
        self.line.remove()
        for kf in self.key_frames:
            kf.remove()

    @property
    def geometries(self):
        """Expose geometries.

        Returns
        -------
        geometries : list
            Empty list (viser artists manage their own handles).
        """
        return list(
            chain(
                self.line.geometries, *[kf.geometries for kf in self.key_frames]
            )
        )


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
        self.radius = radius
        self.A2B = np.asarray(A2B, dtype=float)
        self.resolution = resolution
        self.c = c
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        kwargs = dict(
            name=figure._next_name("sphere"),
            radius=self.radius,
            subdivisions=max(1, self.resolution // 6),
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )
        if self.c is not None:
            kwargs["color"] = _to_viser_color(self.c)
        self._handle = figure.scene.add_icosphere(**kwargs)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the sphere.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
        self.size = np.asarray(size, dtype=float)
        self.A2B = np.asarray(A2B, dtype=float)
        self.c = c
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        kwargs = dict(
            name=figure._next_name("box"),
            dimensions=tuple(self.size),
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )
        if self.c is not None:
            kwargs["color"] = _to_viser_color(self.c)
        self._handle = figure.scene.add_box(**kwargs)

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the box.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
        This parameter is ignored. It is accepted for API compatibility with
        the Open3D backend.

    c : array-like, shape (3,), optional (default: None)
        Color
    """

    def __init__(
        self,
        length=2.0,
        radius=1.0,
        A2B=np.eye(4),
        resolution=20,
        split=4,
        c=None,
    ):
        self.length = length
        self.radius = radius
        self.A2B = np.asarray(A2B, dtype=float)
        self.resolution = resolution
        self.c = c
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        mesh = trimesh.creation.cylinder(
            radius=self.radius,
            height=self.length,
            sections=self.resolution,
        )
        _set_trimesh_color(mesh, self.c)
        self._handle = figure.scene.add_mesh_trimesh(
            name=figure._next_name("cylinder"),
            mesh=mesh,
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the cylinder.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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

    convex_hull : bool, optional (default: False)
        Compute convex hull of mesh.
    """

    def __init__(
        self, filename, A2B=np.eye(4), s=np.ones(3), c=None, convex_hull=False
    ):
        self.filename = filename
        self.A2B = np.asarray(A2B, dtype=float)
        self.s = np.asarray(s, dtype=float)
        self.c = c
        self.convex_hull = convex_hull
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        mesh = trimesh.load(self.filename)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        mesh.apply_scale(self.s)
        if self.convex_hull:
            mesh = mesh.convex_hull
        if self.c is not None:
            _set_trimesh_color(mesh, np.asarray(self.c).ravel()[:3])
        self._handle = figure.scene.add_mesh_trimesh(
            name=figure._next_name("mesh"),
            mesh=mesh,
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the mesh.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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
        self.radii = np.asarray(radii, dtype=float)
        self.A2B = np.asarray(A2B, dtype=float)
        self.resolution = resolution
        self.c = c
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        mesh = trimesh.creation.icosphere(
            subdivisions=max(1, self.resolution // 6), radius=1.0
        )
        mesh.vertices = mesh.vertices * self.radii[np.newaxis]
        _set_trimesh_color(mesh, self.c)
        self._handle = figure.scene.add_mesh_trimesh(
            name=figure._next_name("ellipsoid"),
            mesh=mesh,
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the ellipsoid.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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

    def __init__(
        self, height=1, radius=1, A2B=np.eye(4), resolution=20, c=None
    ):
        self.height = height
        self.radius = radius
        self.A2B = np.asarray(A2B, dtype=float)
        self.resolution = resolution
        self.c = c
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        mesh = trimesh.creation.capsule(
            radius=self.radius,
            height=self.height,
            count=[self.resolution // 2, self.resolution],
        )
        _set_trimesh_color(mesh, self.c)
        self._handle = figure.scene.add_mesh_trimesh(
            name=figure._next_name("capsule"),
            mesh=mesh,
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Start of the capsule's line segment.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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

    def __init__(
        self, height=1, radius=1, A2B=np.eye(4), resolution=20, c=None
    ):
        self.height = height
        self.radius = radius
        self.A2B = np.asarray(A2B, dtype=float)
        self.resolution = resolution
        self.c = c
        self._handle = None

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        mesh = trimesh.creation.cone(
            radius=self.radius,
            height=self.height,
            sections=self.resolution,
        )
        _set_trimesh_color(mesh, self.c)
        self._handle = figure.scene.add_mesh_trimesh(
            name=figure._next_name("cone"),
            mesh=mesh,
            wxyz=_wxyz_from_matrix(self.A2B),
            position=self.A2B[:3, 3],
        )

    def set_data(self, A2B):
        """Update data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Center of the cone's circle.
        """
        self.A2B = np.asarray(A2B, dtype=float)
        if self._handle is None:
            return
        self._handle.wxyz = _wxyz_from_matrix(self.A2B)
        self._handle.position = self.A2B[:3, 3]

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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

    def __init__(
        self,
        normal=np.array([0.0, 0.0, 1.0]),
        d=None,
        point_in_plane=None,
        s=1.0,
        c=None,
    ):
        self.normal = np.asarray(normal, dtype=float)
        self.d = d
        self.point_in_plane = (
            None
            if point_in_plane is None
            else np.asarray(point_in_plane, dtype=float)
        )
        self.s = s
        self.c = c
        self._handle = None
        self._figure = None
        self._name = None

    @staticmethod
    def _resolve_point(normal, d, point_in_plane):
        if point_in_plane is not None:
            return np.asarray(point_in_plane, dtype=float)
        if d is None:
            raise ValueError(
                "Either 'd' or 'point_in_plane' has to be defined!"
            )
        return d * np.asarray(normal, dtype=float)

    def _make_mesh(self):
        point = self._resolve_point(self.normal, self.d, self.point_in_plane)
        x_axis, y_axis = pr.plane_basis_from_normal(self.normal)
        vertices = np.array(
            [
                point + self.s * x_axis + self.s * y_axis,
                point - self.s * x_axis + self.s * y_axis,
                point + self.s * x_axis - self.s * y_axis,
                point - self.s * x_axis - self.s * y_axis,
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [[0, 1, 2], [1, 3, 2], [2, 1, 0], [2, 3, 1]], dtype=np.int32
        )
        return vertices, faces

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        self._figure = figure
        self._name = figure._next_name("plane")
        vertices, faces = self._make_mesh()
        kwargs = dict(
            name=self._name, vertices=vertices, faces=faces, side="double"
        )
        if self.c is not None:
            kwargs["color"] = _to_viser_color(self.c)
        self._handle = figure.scene.add_mesh_simple(**kwargs)

    def set_data(self, normal, d=None, point_in_plane=None, s=None, c=None):
        """Update data.

        Parameters
        ----------
        normal : array-like, shape (3,)
            Plane normal.

        d : float, optional (default: None)
            Distance to origin in Hesse normal form.

        point_in_plane : array-like, shape (3,), optional (default: None)
            Point in plane.

        s : float, optional (default: None)
            Scaling of the plane that will be drawn.

        c : array-like, shape (3,), optional (default: None)
            Color.

        Raises
        ------
        ValueError
            If neither 'd' nor 'point_in_plane' is defined.
        """
        self.normal = np.asarray(normal, dtype=float)
        self.d = d
        self.point_in_plane = (
            None
            if point_in_plane is None
            else np.asarray(point_in_plane, dtype=float)
        )
        if s is not None:
            self.s = s
        if c is not None:
            self.c = c
        if self._figure is None:
            return
        if self._handle is not None:
            self._handle.remove()
        vertices, faces = self._make_mesh()
        kwargs = dict(
            name=self._name, vertices=vertices, faces=faces, side="double"
        )
        if self.c is not None:
            kwargs["color"] = _to_viser_color(self.c)
        self._handle = self._figure.scene.add_mesh_simple(**kwargs)

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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

    def __init__(
        self,
        M,
        cam2world=None,
        virtual_image_distance=1,
        sensor_size=(1920, 1080),
        strict_check=True,
    ):
        self.M = np.asarray(M, dtype=float)
        self.cam2world = (
            np.eye(4)
            if cam2world is None
            else np.asarray(cam2world, dtype=float)
        )
        self.cam2world = pt.check_transform(
            self.cam2world, strict_check=strict_check
        )
        self.virtual_image_distance = virtual_image_distance
        self.sensor_size = sensor_size
        self.strict_check = strict_check
        self._handle = None

    def _compute_segments(self):
        """Compute the 11 line segments that form the camera wireframe.

        The wireframe consists of four frustum rays from the camera centre to
        the corners of the virtual image plane, the rectangle connecting those
        corners, and a small triangle above the top edge that indicates the
        camera's up direction.

        Returns
        -------
        segments : array, shape (11, 2, 3)
            Line segments in world coordinates.
        """
        cam2world = self.cam2world
        focal_length = float(np.mean([self.M[0, 0], self.M[1, 1]]))
        w, h = float(self.sensor_size[0]), float(self.sensor_size[1])
        cx, cy = float(self.M[0, 2]), float(self.M[1, 2])

        corners_in_cam = np.array(
            [
                [0.0 - cx, 0.0 - cy, focal_length],
                [0.0 - cx, h - cy, focal_length],
                [w - cx, h - cy, focal_length],
                [w - cx, 0.0 - cy, focal_length],
            ]
        )
        corners_in_world = pt.transform(
            cam2world, pt.vectors_to_points(corners_in_cam)
        )[:, :3]

        camera_center = cam2world[:3, 3]
        virtual_corners = (
            self.virtual_image_distance
            / focal_length
            * (corners_in_world - camera_center[np.newaxis])
            + camera_center[np.newaxis]
        )

        up = virtual_corners[0] - virtual_corners[1]
        pts = np.array(
            [
                camera_center,
                virtual_corners[0],
                virtual_corners[1],
                virtual_corners[2],
                virtual_corners[3],
                virtual_corners[0] + 0.1 * up,
                0.5 * (virtual_corners[0] + virtual_corners[3]) + 0.5 * up,
                virtual_corners[3] + 0.1 * up,
            ]
        )

        pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (5, 6),
            (6, 7),
            (7, 5),
        ]
        return np.array([[pts[i], pts[j]] for i, j in pairs], dtype=np.float32)

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        segments = self._compute_segments()
        n = len(segments)
        colors = np.zeros((n, 1, 3), dtype=np.uint8)
        self._handle = figure.scene.add_line_segments(
            name=figure._next_name("camera"),
            points=segments,
            colors=colors,
        )

    def set_data(
        self,
        M=None,
        cam2world=None,
        virtual_image_distance=None,
        sensor_size=None,
    ):
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
            self.M = np.asarray(M, dtype=float)
        if cam2world is not None:
            self.cam2world = pt.check_transform(
                cam2world, strict_check=self.strict_check
            )
        if virtual_image_distance is not None:
            self.virtual_image_distance = virtual_image_distance
        if sensor_size is not None:
            self.sensor_size = sensor_size
        if self._handle is None:
            return
        self._handle.points = self._compute_segments()

    def remove(self):
        """Remove artist from figure."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


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

    convex_hull_of_collision_objects : bool, optional (default: False)
        Show convex hull of collision objects.

    s : float, optional (default: 1)
        Scaling of the frames that will be drawn
    """

    def __init__(
        self,
        tm,
        frame,
        show_frames=False,
        show_connections=False,
        show_visuals=False,
        show_collision_objects=False,
        show_name=False,
        whitelist=None,
        convex_hull_of_collision_objects=False,
        s=1.0,
    ):
        self.tm = tm
        self.frame = frame
        self.show_frames = show_frames
        self.show_connections = show_connections
        self.show_visuals = show_visuals
        self.show_collision_objects = show_collision_objects
        self.whitelist = whitelist
        self.convex_hull_of_collision_objects = convex_hull_of_collision_objects
        self.s = s

        if self.frame not in self.tm.nodes:
            raise KeyError("Unknown frame '%s'" % self.frame)

        self.nodes = list(sorted(self.tm._whitelisted_nodes(whitelist)))

        self.frames = {}
        if self.show_frames:
            for node in self.nodes:
                try:
                    node2frame = self.tm.get_transform(node, frame)
                    node_name = node if show_name else None
                    self.frames[node] = Frame(node2frame, node_name, self.s)
                except KeyError:
                    pass

        self.connections = {}
        if self.show_connections:
            for frame_names in self.tm.transforms.keys():
                from_frame, to_frame = frame_names
                if from_frame in self.tm.nodes and to_frame in self.tm.nodes:
                    try:
                        self.tm.get_transform(from_frame, self.frame)
                        self.tm.get_transform(to_frame, self.frame)
                        self.connections[frame_names] = Line3D(np.zeros((2, 3)))
                    except KeyError:
                        pass

        self.visuals = {}
        if show_visuals and hasattr(self.tm, "visuals"):
            self.visuals.update(_objects_to_artists(self.tm.visuals))
        self.collision_objects = {}
        if show_collision_objects and hasattr(self.tm, "collision_objects"):
            self.collision_objects.update(
                _objects_to_artists(
                    self.tm.collision_objects, convex_hull_of_collision_objects
                )
            )

        self.set_data()

    def add_artist(self, figure):
        """Add artist to figure.

        Parameters
        ----------
        figure : Figure
            Figure to which the artist will be added.
        """
        for f in self.frames.values():
            f.add_artist(figure)
        for conn in self.connections.values():
            conn.add_artist(figure)
        for obj in self.visuals.values():
            obj.add_artist(figure)
        for obj in self.collision_objects.values():
            obj.add_artist(figure)

    def set_data(self):
        """Indicate that data has been updated."""
        if self.show_frames:
            for node in self.nodes:
                try:
                    node2frame = self.tm.get_transform(node, self.frame)
                    self.frames[node].set_data(node2frame)
                except KeyError:
                    pass

        if self.show_connections:
            for frame_names, conn in self.connections.items():
                from_frame, to_frame = frame_names
                try:
                    from2ref = self.tm.get_transform(from_frame, self.frame)
                    to2ref = self.tm.get_transform(to_frame, self.frame)
                    points = np.vstack((from2ref[:3, 3], to2ref[:3, 3]))
                    conn.set_data(points)
                except KeyError:
                    pass

        for frame_name, obj in self.visuals.items():
            A2B = self.tm.get_transform(frame_name, self.frame)
            obj.set_data(A2B)

        for frame_name, obj in self.collision_objects.items():
            A2B = self.tm.get_transform(frame_name, self.frame)
            obj.set_data(A2B)

    def remove(self):
        """Remove artist from figure."""
        for f in self.frames.values():
            f.remove()
        for conn in self.connections.values():
            conn.remove()
        for obj in self.visuals.values():
            obj.remove()
        for obj in self.collision_objects.values():
            obj.remove()


def _objects_to_artists(objects, convex_hull=False):
    """Convert geometries from URDF to artists.

    Parameters
    ----------
    objects : list of Geometry
        Objects parsed from URDF.

    convex_hull : bool, optional (default: False)
        Compute convex hull for each object.

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
            # alpha channel is not used
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
                artist = Mesh(
                    obj.filename, s=obj.scale, c=color, convex_hull=convex_hull
                )
            artists[obj.frame] = artist
        except RuntimeError as e:
            warnings.warn(str(e), RuntimeWarning, stacklevel=1)
    return artists
