"""
==============================
Compute Rotor from Two Vectors
==============================

We compute a rotor that rotates one vector to another vector.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
from pytransform3d.plot_utils import plot_vector


random_state = np.random.RandomState(24)
a = random_state.randn(3)
b = random_state.randn(3)
a = pr.norm_vector(a)
b = pr.norm_vector(b)
rotor = pr.rotor_from_two_vectors(a, b)
a_rotated = pr.rotor_apply(rotor, a)

ax = pr.plot_basis()
plot_vector(ax=ax, start=np.zeros(3), direction=a)
plot_vector(ax=ax, start=np.zeros(3), direction=b)
plot_vector(ax=ax, start=np.zeros(3), direction=a_rotated, color="r")
plt.show()
