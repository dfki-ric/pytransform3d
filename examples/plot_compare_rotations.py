import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytransform.rotations import *


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", aspect="equal")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    ax.set_zlim((-2, 2))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plot_basis(ax, np.eye(3))
    axis = 0
    angle = np.pi / 8

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

    p = np.array([-1.0, -1.0, -1.0])
    q = quaternion_from_axis_angle(a)
    R = matrix_from_quaternion(q)
    plot_basis(ax, R, p)

    plt.show()
