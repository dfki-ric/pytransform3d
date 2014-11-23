import numpy as np
import matplotlib.pyplot as plt
from pytransform.rotations import (random_axis_angle, matrix_from_axis_angle,
                                   plot_basis, plot_axis_angle)


original = random_axis_angle(np.random.RandomState(0))
ax = plot_axis_angle(a=original)
for fraction in np.linspace(0, 1, 50):
    a = original.copy()
    a[-1] = fraction * original[-1]
    R = matrix_from_axis_angle(a)
    plot_basis(ax, R, alpha=0.2)
plt.show()
