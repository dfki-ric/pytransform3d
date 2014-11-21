import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform.rotations import *


def update_lines(t, start, end, n_frames, lines):
    progress = t / float(n_frames - 1)
    if progress < 0.5:
        q = start + 2.0 * progress * (end - start)
    else:
        q = end + (2.0 * progress - 1.0) * (start - end)
    print start, end, q
    R = matrix_from_quaternion(q)

    lines[0].set_data([0, R[0, 0]], [0, R[1, 0]])
    lines[0].set_3d_properties([0, R[2, 0]])

    lines[1].set_data([0, R[0, 1]], [0, R[1, 1]])
    lines[1].set_3d_properties([0, R[2, 1]])

    lines[2].set_data([0, R[0, 2]], [0, R[1, 2]])
    lines[2].set_3d_properties([0, R[2, 2]])

    test = R.dot(np.ones(3) / np.sqrt(3.0))
    lines[3].set_data([test[0] / 2.0, test[0]], [test[1] / 2.0, test[1]])
    lines[3].set_3d_properties([test[2] / 2.0, test[2]])

    return lines


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d", aspect="equal")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    lines = [ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],
             ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
             ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],
             ax.plot([0, 1], [0, 1], [0, 1], c="gray", lw=3)[0]]

    # Generate random start and goal
    np.random.seed(3)
    start = quaternion_from_angle_axis(np.array([np.pi, 0, 0, 0]))
    start[1:] = np.random.randn(3)
    end = quaternion_from_angle_axis(np.array([np.pi, 0, 0, 0]))
    end[1:] = np.random.randn(3)
    n_frames = 100

    anim = animation.FuncAnimation(fig, update_lines, n_frames,
                                   fargs=(start, end, n_frames, lines),
                                   interval=50, blit=False)

    plt.show()
