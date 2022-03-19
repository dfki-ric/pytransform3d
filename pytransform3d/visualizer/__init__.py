"""Optional 3D renderer based on Open3D's visualizer."""
import warnings
try:
    import open3d as o3d
    from ._artists import (Artist, Line3D, PointCollection3D, Vector3D, Frame,
                           Trajectory, Camera, Box, Sphere, Cylinder, Mesh,
                           Ellipsoid, Capsule, Cone, Plane, Graph)
    from ._figure import figure, Figure

    __all__ = ["figure", "Figure", "Artist", "Line3D", "PointCollection3D",
               "Vector3D", "Frame", "Trajectory", "Camera", "Box", "Sphere",
               "Cylinder", "Mesh", "Ellipsoid", "Capsule", "Cone", "Plane",
               "Graph"]
except ImportError:
    warnings.warn("3D visualizer is not available. Install open3d.")
