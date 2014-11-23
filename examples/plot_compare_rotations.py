import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform.rotations import *


ax = plot_basis(R=np.eye(3), ax_s=2)
axis = 0
angle = np.pi / 2

p = np.array([1.0, 1.0, 1.0])
euler = [0, 0, 0]
euler[axis] = angle
R = matrix_from_euler_xyz(euler)
plot_basis(ax, R, p)

p = np.array([1.0, -1.0, 1.0])
euler = [0, 0, 0]
euler[2 - axis] = angle
R = matrix_from_euler_zyx(euler)
plot_basis(ax, R, p)

p = np.array([1.0, 1.0, -1.0])
R = matrix_from_angle(axis, angle)
plot_basis(ax, R, p)

p = np.array([1.0, -1.0, -1.0])
e = [unitx, unity, unitz][axis]
a = np.hstack((e, (angle,)))
R = matrix_from_axis_angle(a)
plot_basis(ax, R, p)
plot_axis_angle(ax, a, p, s=0.5)

p = np.array([-1.0, -1.0, -1.0])
q = quaternion_from_axis_angle(a)
R = matrix_from_quaternion(q)
plot_basis(ax, R, p)

plt.show()
