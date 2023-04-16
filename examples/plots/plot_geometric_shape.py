"""TODO"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import pytransform3d.geometry as pg
import pytransform3d.plot_utils as ppu


#shape = pg.Sphere(np.eye(4), 1.0)
#shape = pg.Ellipsoid(np.eye(4), [0.2, 0.3, 0.5])
#shape = pg.Capsule(np.eye(4), 1.2, 0.2)
#shape = pg.Cylinder(np.eye(4), 0.2, 1.2)
#shape = pg.Cone(np.eye(4), 1.2, 0.2)
shape = pg.Box(np.eye(4), [1.5, 2.0, 0.9])

plt.figure()
ax = ppu.make_3d_axis(1)
x, y, z = shape.surface(20)
#ax.plot_surface(x, y, z, alpha=0.5, linewidth=0)
#ax.plot_wireframe(x, y, z, alpha=0.5)
vertices, triangles = shape.mesh()
"""
import pytransform3d.visualizer as pv
import open3d as o3d
fig = pv.figure()
mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(triangles))
fig.add_geometry(mesh)
fig.show()
#"""
vectors = np.array([vertices[[i, j, k]] for i, j, k in triangles])
surface = Line3DCollection(vectors)
ax.add_collection3d(surface)
surface = Poly3DCollection(vectors)
surface.set_alpha(0.1)
ax.add_collection3d(surface)
plt.show()
