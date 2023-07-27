import numpy as np


def load_mesh(filename, convex_hull=False, return_as_open3d_mesh=False):
    vertices = np.empty((0, 3), dtype=float)
    triangles = np.empty((0, 3), dtype=int)
    done = False

    try:
        import trimesh
        geometry = trimesh.load(filename)
        done = True

        if isinstance(geometry, trimesh.Scene):
            geometry = trimesh.util.concatenate(
                list(geometry.geometry.values()))
        if convex_hull:
            geometry = geometry.convex_hull

        vertices = geometry.vertices
        triangles = geometry.faces

        if return_as_open3d_mesh:
            import open3d
            return open3d.geometry.TriangleMesh(
                open3d.utility.Vector3dVector(vertices),
                open3d.utility.Vector3iVector(triangles))
    except ImportError:
        pass

    try:
        import open3d
        mesh = open3d.io.read_triangle_mesh(filename)
        done = True

        if convex_hull:
            mesh = mesh.compute_convex_hull()[0]

        if return_as_open3d_mesh:
            return mesh
        else:
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
    except ImportError:
        pass

    if not done:
        raise IOError("Could not load mesh from '%s'" % filename)

    return vertices, triangles
