import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform.rotations import *


velocity = None
last_R = None


def update_lines(step, start, end, n_frames, rot, profile, slerp=True):
    global velocity
    global last_R

    if step == 0:
        velocity = []
        last_R = matrix_from_quaternion(start)

    if step <= n_frames / 2:
        t = step / float(n_frames / 2 - 1)
        w1, w2 = 1 - t, t
        if slerp:
            omega = np.arccos(start.dot(end))
            w1 = np.sin((1 - t) * omega) / np.sin(omega)
            w2 = np.sin(t * omega) / np.sin(omega)
        q = w1 * start + w2 * end
    else:
        t = (step - n_frames / 2) / float(n_frames / 2 - 1)
        w1, w2 = 1 - t, t
        if slerp:
            omega = np.arccos(start.dot(end))
            w1 = np.sin((1 - t) * omega) / np.sin(omega)
            w2 = np.sin(t * omega) / np.sin(omega)
        q = w1 * end + w2 * start

    print step, t, start, end, q
    R = matrix_from_quaternion(q)

    rot[0].set_data([0, R[0, 0]], [0, R[1, 0]])
    rot[0].set_3d_properties([0, R[2, 0]])

    rot[1].set_data([0, R[0, 1]], [0, R[1, 1]])
    rot[1].set_3d_properties([0, R[2, 1]])

    rot[2].set_data([0, R[0, 2]], [0, R[1, 2]])
    rot[2].set_3d_properties([0, R[2, 2]])

    test = R.dot(np.ones(3) / np.sqrt(3.0))
    rot[3].set_data([test[0] / 2.0, test[0]], [test[1] / 2.0, test[1]])
    rot[3].set_3d_properties([test[2] / 2.0, test[2]])

    velocity.append(np.linalg.norm(R - last_R))
    last_R = R
    profile.set_data(np.linspace(0, 1, len(velocity)), velocity)

    return rot


if __name__ == "__main__":
    # Generate random start and goal
    np.random.seed(3)
    start = np.array([np.pi, 0, 0, 0])
    start[1:] = np.random.randn(3)
    start = quaternion_from_angle_axis(start)
    end = np.array([np.pi, 0, 0, 0])
    end[1:] = np.random.randn(3)
    end = quaternion_from_angle_axis(end)
    n_frames = 100

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(121, projection="3d", aspect="equal")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    rot = [ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],
           ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
           ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],
           ax.plot([0, 1], [0, 1], [0, 1], c="gray", lw=3)[0]]

    ax = fig.add_subplot(122)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    profile = ax.plot(0, 0)[0]

    anim = animation.FuncAnimation(fig, update_lines, n_frames,
                                   fargs=(start, end, n_frames, rot, profile),
                                   interval=50, blit=False)

    plt.show()
