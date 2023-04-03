"""
==========================================
Construct Rotation Matrix from Two Vectors
==========================================

We compute rotation matrix from two vectors that form a plane. The x-axis will
point in the same direction as the first vector, the y-axis corresponds to the
normalized vector rejection of b on a, and the z-axis is the cross product of
the other basis vectors.
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.rotations import (
    matrix_from_two_vectors, plot_basis, random_vector)
from pytransform3d.plot_utils import plot_vector


rng = np.random.default_rng(1)
a = random_vector(rng, 3) * 0.3
b = random_vector(rng, 3) * 0.3
R = matrix_from_two_vectors(a, b)

ax = plot_vector(direction=a, color="r")
plot_vector(ax=ax, direction=b, color="g")
plot_basis(ax=ax, R=R)
plt.show()
