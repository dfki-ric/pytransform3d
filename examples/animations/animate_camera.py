"""
================
Animate Camera
================

Animate a camera moving along a circular trajectory while looking at a target.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from pytransform3d.rotations import matrix_from_euler
from pytransform3d.transformations import transform_from
from pytransform3d.plot_utils import Frame, Camera, make_3d_axis


def update_camera(step, n_frames, camera):
    phi = 2 * step / n_frames * np.pi
    tf = transform_from(
        matrix_from_euler([-1 / 2 * np.pi, phi, 0], 0, 1, 2, False),
        -10 * np.array([np.sin(phi), np.cos(phi), 0]),
    )
    camera.set_data(tf)


if __name__ == "__main__":
    n_frames = 50

    fig = plt.figure(figsize=(5, 5))
    ax = make_3d_axis(15)

    frame = Frame(np.eye(4), label="target", s=3, draw_label_indicator=False)
    frame.add_frame(ax)

    fl = 3000  # [pixels]
    w, h = 1920, 1080  # [pixels]
    M = np.array(((fl, 0, w // 2), (0, fl, h // 2), (0, 0, 1)))
    camera = Camera(M, np.eye(4), virtual_image_distance=5, sensor_size=(w, h), c="c")
    camera.add_camera(ax)

    anim = animation.FuncAnimation(
        fig, update_camera, n_frames, fargs=(n_frames, camera), interval=50, blit=False
    )

    plt.show()
