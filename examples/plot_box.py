"""
========
Plot Box
========
"""
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import plot_box


from pytransform3d.rotations import unitx, unitz, perpendicular_to_vectors, norm_vector
from mpl_toolkits.mplot3d.art3d import Text3D
def plot_length_variable(ax, start, end, name, color="k", **kwargs):  # TODO move to plot_utils
    direction = end - start
    length = np.linalg.norm(direction)
    mid1 = start + 0.4 * direction
    mid2 = start + 0.6 * direction
    mid = start + 0.45 * direction

    ax.plot([start[0], mid1[0]], [start[1], mid1[1]], [start[2], mid1[2]], color=color)
    ax.plot([end[0], mid2[0]], [end[1], mid2[1]], [end[2], mid2[2]], color=color)

    if np.linalg.norm(direction / length - unitz) < np.finfo(float).eps:
        axis = unitx
    else:
        axis = unitz

    mark = norm_vector(perpendicular_to_vectors(direction, axis)) * 0.03 * length
    mark_start1 = start + mark
    mark_start2 = start - mark
    mark_end1 = end + mark
    mark_end2 = end - mark
    ax.plot([mark_start1[0], mark_start2[0]],
            [mark_start1[1], mark_start2[1]],
            [mark_start1[2], mark_start2[2]],
            color=color)
    ax.plot([mark_end1[0], mark_end2[0]],
            [mark_end1[1], mark_end2[1]],
            [mark_end1[2], mark_end2[2]],
            color=color)
    text = Text3D(mid[0], mid[1], mid[2], text=name, zdir="x", **kwargs)
    ax._add_text(text)

    return ax


random_state = np.random.RandomState(42)
ax = plot_box(size=[1, 1, 1], wireframe=False, alpha=0.1, color="k")
plot_box(ax=ax, size=[1, 1, 1], wireframe=True, alpha=0.3)
plot_length_variable(
    ax=ax,
    start=np.array([-0.5, -0.5, 0.55]), end=np.array([0.5, -0.5, 0.55]),
    name="a",
    fontsize=14, fontfamily="serif")
plot_length_variable(
    ax=ax,
    start=np.array([0.55, -0.5, 0.5]), end=np.array([0.55, 0.5, 0.5]),
    name="b",
    fontsize=14, fontfamily="serif")
plot_length_variable(
    ax=ax,
    start=np.array([-0.55, -0.5, -0.5]), end=np.array([-0.55, -0.5, 0.5]),
    name="c",
    fontsize=14, fontfamily="serif")
plt.show()