import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis


plt.figure()
ax = make_3d_axis(1)
plt.setp(ax, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), zlim=(-0.05, 1.05),
        xlabel="X", ylabel="Y", zlabel="Z")

basis = np.eye(3)
for d, c in enumerate(["r", "g", "b"]):
    ax.plot([0.0, basis[0, d]],
            [0.0, basis[1, d]],
            [0.0, basis[2, d]], color=c, lw=5)

plt.show()