import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform3d.rotations import *


ax = plot_basis(R=np.eye(3), p=np.array([-1.0, 0.0, 0.0]), ax_s=2)

R1 = matrix_from_angle(0, np.pi / 4.0)
R2 = matrix_from_angle(2, np.pi / 2.0)

plot_basis(ax, R1, np.array([1.0, 0.0, 0.0]))
plot_basis(ax, R1.dot(R2), np.array([1.0, 1.5, 0.0]))
plot_basis(ax, R2.dot(R1), np.array([1.0, -1.5, 0.0]))

ax.view_init(azim=10, elev=25)

plt.show()