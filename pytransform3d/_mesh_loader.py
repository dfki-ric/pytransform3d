"""Common interface to load meshes."""
import abc

import numpy as np


def load_mesh(filename):
    """Load mesh from file.

    This feature relies on optional dependencies. It can use trimesh or
    Open3D to load meshes. If both are not available, it will fail.
    Furthermore, some mesh formats require additional dependencies. For
    example, loading collada files ('.dae' file ending) requires pycollada
    and trimesh.

    Parameters
    ----------
    filename : str
        File in which the mesh is stored.

    Returns
    -------
    mesh : MeshBase
        Mesh instance.
    """
    mesh = _Trimesh(filename)
    loader_available = mesh.load()

    if not loader_available:  # pragma: no cover
        mesh = _Open3DMesh(filename)
        loader_available = mesh.load()

    if not loader_available:  # pragma: no cover
        raise ImportError(
            "Could not load mesh from '%s'. Please install one of the "
            "optional dependencies 'trimesh' or 'open3d'." % filename)

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
        loader_available : bool
            Is the mesh loader available?
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
        if isinstance(obj, trimesh.Scene):  # pragma: no cover
            obj = self._convert_scene_to_mesh(obj)
        self.mesh = obj
        return True

    def _convert_scene_to_mesh(self, obj):  # pragma: no cover
        # Special case in which we load a collada file that contains
        # multiple meshes. We might lose textures. This is excluded
        # from testing as it would add another dependency.
        import trimesh
        trimesh_version_parts = trimesh.__version__.split(".")
        major_version = int(trimesh_version_parts[0])
        if major_version >= 4:
            try:
                minor_version = int(trimesh_version_parts[1])
            except:
                minor_version = 0
            try:
                patch_version = int(trimesh_version_parts[2])
            except:  # most likely release candidate (rc) version
                patch_version = 0
            if minor_version >= 4 and patch_version >= 9:
                return obj.to_mesh()
        return obj.dump(concatenate=True)

    def convex_hull(self):
        self.mesh = self.mesh.convex_hull

    def get_open3d_mesh(self):  # pragma: no cover
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


class _Open3DMesh(MeshBase):  # pragma: no cover
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
