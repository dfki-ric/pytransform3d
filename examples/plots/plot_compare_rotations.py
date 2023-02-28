"""
========================================
Compare Various Definitions of Rotations
========================================
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d import rotations as pr


ax = pr.plot_basis(R=np.eye(3), ax_s=2, lw=3)
axis = 2
angle = np.pi

p = np.array([1.0, 1.0, 1.0])
euler = [0, 0, 0]
euler[axis] = angle
R = pr.active_matrix_from_intrinsic_euler_xyz(euler)
pr.plot_basis(ax, R, p)

p = np.array([-1.0, 1.0, 1.0])
euler = [0, 0, 0]
euler[2 - axis] = angle
R = pr.active_matrix_from_intrinsic_euler_zyx(euler)
pr.plot_basis(ax, R, p)

p = np.array([1.0, 1.0, -1.0])
R = pr.active_matrix_from_angle(axis, angle)
pr.plot_basis(ax, R, p)

p = np.array([1.0, -1.0, -1.0])
e = [pr.unitx, pr.unity, pr.unitz][axis]
a = np.hstack((e, (angle,)))
R = pr.matrix_from_axis_angle(a)
pr.plot_basis(ax, R, p)
pr.plot_axis_angle(ax, a, p)

p = np.array([-1.0, -1.0, -1.0])
q = pr.quaternion_from_axis_angle(a)
R = pr.matrix_from_quaternion(q)
pr.plot_basis(ax, R, p)

plt.show()
