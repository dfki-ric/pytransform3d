import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d


plt.figure()
ax = plt.subplot(111, projection="3d", aspect="equal")
plt.setp(ax, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), zlim=(-1.05, 0.05),
         xlabel="X", ylabel="Y", zlabel="Z")

basis = np.eye(3)
basis[:, 2] *= -1.0
for d, c in enumerate(["r", "g", "b"]):
    ax.plot([0.0, basis[0, d]],
            [0.0, basis[1, d]],
            [0.0, basis[2, d]], color=c, lw=5)

plt.show()