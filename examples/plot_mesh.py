"""
=========
Plot Mesh
=========

This example shows how to load an STL mesh. This example must be
run from within the main folder because it uses a
hard-coded path to the STL file.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import plot_mesh


ax = plot_mesh(filename="test/test_data/cone.stl", s=5 * np.ones(3), alpha=0.3)
plot_transform(ax=ax, A2B=np.eye(4), s=0.3, lw=3)
plt.show()