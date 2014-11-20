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
    alpha, beta, gamma = np.pi, np.pi, np.pi

    p = np.array([0.6, 0.4, 0.4])
    R = matrix_from_euler_xyz([alpha, 0, 0])
    plot_basis(ax, R, p)
    R = matrix_from_euler_xyz([alpha, beta, 0])
    plot_basis(ax, R, 2 * p)
    R = matrix_from_euler_xyz([alpha, beta, gamma])
    plot_basis(ax, R, 3 * p)

    p = np.array([0.4, 0.6, 0.4])
    R = matrix_from_euler_zyx([alpha, 0, 0])
    plot_basis(ax, R, p, alpha=0.5)
    R = matrix_from_euler_zyx([alpha, beta, 0])
    plot_basis(ax, R, 2 * p, alpha=0.5)
    R = matrix_from_euler_zyx([alpha, beta, gamma])
    plot_basis(ax, R, 3 * p, alpha=0.5)

    plt.show()
