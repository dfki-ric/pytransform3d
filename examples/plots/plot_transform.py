"""
===================
Plot Transformation
===================

We can display transformations by plotting the basis vectors of the
corresponding coordinate frame.
"""
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis

ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
plot_transform(ax=ax)
plt.tight_layout()
plt.show()
