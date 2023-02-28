"""
=======================
Transform Concatenation
=======================

In this example, we have a point p that is defined in a frame C, we know
the transform C2B and B2A. We can construct a transform C2A to extract the
position of p in frame A.
"""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pyrot
import pytransform3d.transformations as pytr


p = np.array([0.0, 0.0, -0.5])
a = np.array([0.0, 0.0, 1.0, np.pi])
B2A = pytr.transform_from(pyrot.matrix_from_axis_angle(a), p)

p = np.array([0.3, 0.4, 0.5])
a = np.array([0.0, 0.0, 1.0, -np.pi / 2.0])
C2B = pytr.transform_from(pyrot.matrix_from_axis_angle(a), p)

C2A = pytr.concat(C2B, B2A)
p = pytr.transform(C2A, np.ones(4))

ax = pytr.plot_transform(A2B=B2A)
pytr.plot_transform(ax, A2B=C2A)
ax.scatter(p[0], p[1], p[2])
plt.show()
