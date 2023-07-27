import abc

import numpy as np


def load_mesh(filename, convex_hull=False):
    """Load mesh from file.

    Parameters
    ----------
    filename : str
        File in which the mesh is stored.

    convex_hull : bool, optional (default: False)
        Compute convex hull of mesh.

    Returns
    -------
    mesh : MeshBase
        Mesh instance.
    """
    mesh = _Trimesh(filename)
    success = mesh.load()

    if not success:
        mesh = _Open3DMesh(filename)
        success = mesh.load()

    if not success:
        raise IOError("Could not load mesh from '%s'" % filename)

    if convex_hull:
        mesh.convex_hull()

    return mesh


class MeshBase(abc.ABC):
    """Abstract base class of meshes.

    Parameters
    ----------
    filename : str
        File in which the mesh is stored.
    """
    def __init__(self, filename):
        self.filename = filename

    @abc.abstractmethod
    def load(self):
        """Load mesh from file.

        Returns
        -------
        success : bool
            Has the mesh been loaded?
        """

    @abc.abstractmethod
    def convex_hull(self):
        """Compute convex hull of mesh."""

    @abc.abstractmethod
    def get_open3d_mesh(self):
        """Return Open3D mesh.

        Returns
        -------
        mesh : open3d.geometry.TriangleMesh
            Open3D mesh.
        """

    @property
    @abc.abstractmethod
    def vertices(self):
        """Vertices."""

    @property
    @abc.abstractmethod
    def triangles(self):
        """Triangles."""


class _Trimesh(MeshBase):
    def __init__(self, filename):
        super(_Trimesh, self).__init__(filename)
        self.mesh = None

    def load(self):
        try:
            import trimesh
        except ImportError:
            return False

        obj = trimesh.load(self.filename)
        if isinstance(obj, trimesh.Scene):
            obj = obj.dump().sum()
        self.mesh = obj
        return True

    def convex_hull(self):
        self.mesh = self.mesh.convex_hull

    def get_open3d_mesh(self):
        import open3d
        return open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(self.vertices),
            open3d.utility.Vector3iVector(self.triangles))

    @property
    def vertices(self):
        return self.mesh.vertices

    @property
    def triangles(self):
        return self.mesh.faces


class _Open3DMesh(MeshBase):
    def __init__(self, filename):
        super(_Open3DMesh, self).__init__(filename)
        self.mesh = None

    def load(self):
        try:
            import open3d
        except ImportError:
            return False
        self.mesh = open3d.io.read_triangle_mesh(self.filename)
        return True

    def convex_hull(self):
        assert self.mesh is not None
        self.mesh = self.mesh.compute_convex_hull()[0]

    def get_open3d_mesh(self):
        return self.mesh

    @property
    def vertices(self):
        return np.asarray(self.mesh.vertices)

    @property
    def triangles(self):
        return np.asarray(self.mesh.triangles)