import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import transform, plot_transform
from pytransform3d.plot_utils import make_3d_axis, Arrow3D


plt.figure()
ax = make_3d_axis(1)
plt.setp(ax, xlim=(-1.05, 1.05), ylim=(-0.55, 1.55), zlim=(-1.05, 1.05),
            xlabel="X", ylabel="Y", zlabel="Z")
ax.view_init(elev=90, azim=-90)
ax.set_xticks(())
ax.set_yticks(())
ax.set_zticks(())

random_state = np.random.RandomState(42)
PA = np.ones((10, 4))
PA[:, :3] = 0.1 * random_state.randn(10, 3)
PA[:, 0] += 0.3
PA[:, :3] += 0.3

x_translation = -0.1
y_translation = 0.2
z_rotation = np.pi / 4.0
A2B = np.array([
    [np.cos(z_rotation), -np.sin(z_rotation), 0.0, x_translation],
    [np.sin(z_rotation), np.cos(z_rotation), 0.0, y_translation],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
PB = transform(A2B, PA)

plot_transform(ax=ax, A2B=np.eye(4))
ax.scatter(PA[:, 0], PA[:, 1], PA[:, 2], c="orange")
plot_transform(ax=ax, A2B=A2B, ls="--", alpha=0.5)
ax.scatter(PB[:, 0], PB[:, 1], PB[:, 2], c="cyan")

axis_arrow = Arrow3D(
    [0.7, 0.3],
    [0.4, 0.9],
    [0.2, 0.2],
    mutation_scale=20, lw=3, arrowstyle="-|>", color="k")
ax.add_artist(axis_arrow)

plt.tight_layout()
plt.show()