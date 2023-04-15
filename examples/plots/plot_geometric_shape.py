"""TODO"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.geometry as pg
import pytransform3d.plot_utils as ppu


#shape = pg.Sphere(np.eye(4), 1.0)
#shape = pg.Ellipsoid(np.eye(4), [0.2, 0.3, 0.5])
#shape = pg.Capsule(np.eye(4), 1.2, 0.2)
shape = pg.Cylinder(np.eye(4), 0.2, 1.2)

plt.figure()
ax = ppu.make_3d_axis(1)
x, y, z = shape.surface(20)
ax.plot_surface(x, y, z, alpha=0.5, linewidth=0)
ax.plot_wireframe(x, y, z, alpha=0.5)
plt.show()
