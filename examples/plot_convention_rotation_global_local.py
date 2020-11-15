"""
================================================================
Convention for Rotation: Passive / Active, Extrinsic / Intrinsic
================================================================

We will compare all possible combinations of passive and active
rotations and extrinsic and intrinsic concatenation of rotations.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from pytransform3d.rotations import passive_matrix_from_angle, active_matrix_from_angle, plot_basis

plt.figure(figsize=(8, 8))
plt.subplots_adjust(left=0, right=1, bottom=0.05, top=0.95, wspace=0, hspace=0.3)
axes = [plt.subplot(4, 3, 1 + i, projection="3d") for i in range(12)]
for i in range(len(axes)):
    plt.setp(axes[i], xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
            xlabel="X", ylabel="Y", zlabel="Z")
    axes[i].view_init(elev=20, azim=60)

Rx45 = passive_matrix_from_angle(0, np.deg2rad(45))
Rz45 = passive_matrix_from_angle(2, np.deg2rad(45))

axes[0].set_title("Passive Extrinsic Rotations", y=0.95)
plot_basis(ax=axes[0], R=np.eye(3))
axes[1].set_title("$R_x(45^{\circ})$", y=0.95)
plot_basis(ax=axes[1], R=Rx45)
axes[2].set_title("$R_z(45^{\circ}) R_x(45^{\circ})$", y=0.95)
plot_basis(ax=axes[2], R=Rz45.dot(Rx45))
axes[3].set_title("Passive Intrinsic Rotations", y=0.95)
plot_basis(ax=axes[3], R=np.eye(3))
axes[4].set_title("$R_x(45^{\circ})$", y=0.95)
plot_basis(ax=axes[4], R=Rx45)
axes[5].set_title("$R_x(45^{\circ}) R_{z'}(45^{\circ})$", y=0.95)
plot_basis(ax=axes[5], R=Rx45.dot(Rz45))

Rx45 = active_matrix_from_angle(0, np.deg2rad(45))
Rz45 = active_matrix_from_angle(2, np.deg2rad(45))

axes[6].set_title("Active Extrinsic Rotations", y=0.95)
plot_basis(ax=axes[6], R=np.eye(3))
axes[7].set_title("$R_x(45^{\circ})$", y=0.95)
plot_basis(ax=axes[7], R=Rx45)
axes[8].set_title("$R_z(45^{\circ}) R_x(45^{\circ})$", y=0.95)
plot_basis(ax=axes[8], R=Rz45.dot(Rx45))
axes[9].set_title("Active Intrinsic Rotations", y=0.95)
plot_basis(ax=axes[9], R=np.eye(3))
axes[10].set_title("$R_x(45^{\circ})$", y=0.95)
plot_basis(ax=axes[10], R=Rx45)
axes[11].set_title("$R_x(45^{\circ}) R_{z'}(45^{\circ})$", y=0.95)
plot_basis(ax=axes[11], R=Rx45.dot(Rz45))

plt.show()
