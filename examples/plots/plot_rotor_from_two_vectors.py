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
rotor = pr.rotor_from_two_vectors(pr.norm_vector(a), pr.norm_vector(b))
a_rotated = pr.rotor_apply(rotor, a)
axis = pr.norm_vector(np.cross(a, b))
angle = pr.angle_between_vectors(a, b)

ax = pr.plot_basis()
plot_vector(ax=ax, start=np.zeros(3), direction=a, color="#aa0000")
plot_vector(ax=ax, start=np.zeros(3), direction=b, color="#00aa00")
plot_vector(ax=ax, start=np.zeros(3), direction=a_rotated, color="#0000aa")
pr.plot_axis_angle(ax=ax, a=np.r_[axis, angle])
plt.show()
