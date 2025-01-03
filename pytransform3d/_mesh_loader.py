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
    # since trimesh does not support color for STL files, we try Open3D first
    if filename.endswith(".stl"):
        mesh = _Open3DMesh(filename)
        mesh_loaded = mesh.load()
    else:  # pragma: no cover
        mesh_loaded = False

    # trimesh is usually better for other formats
    if not mesh_loaded:  # pragma: no cover
        mesh = _Trimesh(filename)
        mesh_loaded = mesh.load()

    if not mesh_loaded:  # pragma: no cover
        mesh = _Open3DMesh(filename)
        mesh_loaded = mesh.load()

    if not mesh_loaded:  # pragma: no cover
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
        self.mesh = trimesh.load_mesh(self.filename)
        return True

    def convex_hull(self):
        self.mesh = self.mesh.convex_hull

    def get_open3d_mesh(self):  # pragma: no cover
        import open3d
        import trimesh
        if isinstance(self.mesh, trimesh.Scene):
            mesh = open3d.geometry.TriangleMesh()
            for d in self.mesh.dump():
                if isinstance(d, trimesh.Trimesh):
                    mesh += self._trimesh_to_open3d_mesh(d)
        else:  # self.mesh is an instance of trimesh.Trimesh
            mesh = self._trimesh_to_open3d_mesh(self.mesh)
        return mesh

    def _trimesh_to_open3d_mesh(self, tri_mesh):  # pragma: no cover
        import open3d
        import trimesh
        mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(tri_mesh.vertices),
            open3d.utility.Vector3iVector(tri_mesh.faces))
        if isinstance(tri_mesh.visual, trimesh.visual.ColorVisuals):
            mesh.vertex_colors = open3d.utility.Vector3dVector(
                tri_mesh.visual.vertex_colors[:, :3] / 255.0)
        return mesh

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
