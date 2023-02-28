"""
===========================================
Interpolate Between Quaternion Orientations
===========================================

We can interpolate between two orientations that are represented by quaternions
either linearly or with slerp (spherical linear interpolation).
Here we compare both methods and measure the angular velocity between two
successive steps. We can see that linear interpolation results in a
non-constant angular velocity. Usually it is a better idea to interpolate with
slerp.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
from pytransform3d import rotations as pr


velocity = None
last_R = None


def interpolate_linear(start, end, t):
    return (1 - t) * start + t * end


def update_lines(step, start, end, n_frames, rot, profile):
    global velocity
    global last_R

    if step == 0:
        velocity = []
        last_R = pr.matrix_from_quaternion(start)

    if step <= n_frames / 2:
        t = step / float(n_frames / 2 - 1)
        q = pr.quaternion_slerp(start, end, t)
    else:
        t = (step - n_frames / 2) / float(n_frames / 2 - 1)
        q = interpolate_linear(end, start, t)

    R = pr.matrix_from_quaternion(q)

    # Draw new frame
    rot[0].set_data(np.array([0, R[0, 0]]), [0, R[1, 0]])
    rot[0].set_3d_properties([0, R[2, 0]])

    rot[1].set_data(np.array([0, R[0, 1]]), [0, R[1, 1]])
    rot[1].set_3d_properties([0, R[2, 1]])

    rot[2].set_data(np.array([0, R[0, 2]]), [0, R[1, 2]])
    rot[2].set_3d_properties([0, R[2, 2]])

    # Update vector in frame
    test = R.dot(np.ones(3) / np.sqrt(3.0))
    rot[3].set_data(
        np.array([test[0] / 2.0, test[0]]), [test[1] / 2.0, test[1]])
    rot[3].set_3d_properties([test[2] / 2.0, test[2]])

    velocity.append(np.linalg.norm(R - last_R))
    last_R = R
    profile.set_data(np.linspace(0, 1, n_frames)[:len(velocity)], velocity)

    return rot


if __name__ == "__main__":
    # Generate random start and goal
    np.random.seed(3)
    start = np.array([0, 0, 0, np.pi])
    start[:3] = np.random.randn(3)
    start = pr.quaternion_from_axis_angle(start)
    end = np.array([0, 0, 0, np.pi])
    end[:3] = np.random.randn(3)
    end = pr.quaternion_from_axis_angle(end)
    n_frames = 200

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlim((-1, 1))
    ax.set_ylim((-1, 1))
    ax.set_zlim((-1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    Rs = pr.matrix_from_quaternion(start)
    Re = pr.matrix_from_quaternion(end)

    rot = [ax.plot([0, 1], [0, 0], [0, 0], c="r", lw=3)[0],
           ax.plot([0, 0], [0, 1], [0, 0], c="g", lw=3)[0],
           ax.plot([0, 0], [0, 0], [0, 1], c="b", lw=3)[0],
           ax.plot([0, 1], [0, 1], [0, 1], c="gray", lw=3)[0],

           ax.plot([0, Rs[0, 0]], [0, Rs[1, 0]], [0, Rs[2, 0]], c="r", lw=3,
                   alpha=0.5)[0],
           ax.plot([0, Rs[0, 1]], [0, Rs[1, 1]], [0, Rs[2, 1]], c="g", lw=3,
                   alpha=0.5)[0],
           ax.plot([0, Rs[0, 2]], [0, Rs[1, 2]], [0, Rs[2, 2]], c="b", lw=3,
                   alpha=0.5)[0],

           ax.plot([0, Re[0, 0]], [0, Re[1, 0]], [0, Re[2, 0]], c="orange",
                   lw=3, alpha=0.5)[0],
           ax.plot([0, Re[0, 1]], [0, Re[1, 1]], [0, Re[2, 1]], c="turquoise",
                   lw=3, alpha=0.5)[0],
           ax.plot([0, Re[0, 2]], [0, Re[1, 2]], [0, Re[2, 2]], c="violet",
                   lw=3, alpha=0.5)[0]]

    ax = fig.add_subplot(122)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    profile = ax.plot(0, 0)[0]

    anim = animation.FuncAnimation(fig, update_lines, n_frames,
                                   fargs=(start, end, n_frames, rot, profile),
                                   interval=50, blit=False)

    plt.show()
