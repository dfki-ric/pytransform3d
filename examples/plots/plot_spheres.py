import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import plot_sphere, plot_spheres
import time

time_single = []
time_multi = []
speedup = []

print("n_spheres", "single", "\t", "multi", "\t", "speedup", sep="\t")

for n_spheres in range(51):
    random_state = np.random.RandomState(0)
    P = 2 * random_state.rand(n_spheres, 3) - 1
    radii = random_state.rand(n_spheres) / 2
    colors = random_state.rand(n_spheres, 3)
    alphas = random_state.rand(n_spheres)

    for multi in range(2):
        start = time.time()

        if multi:
            plot_spheres(p=P, radius=radii, color=colors, alpha=alphas, wireframe=False)

        else:
            for p, radius, color, alpha in zip(P, radii, colors, alphas):
                plot_sphere(p=p, radius=radius, color=color, alpha=alpha, wireframe=False)

        end = time.time()
        diff = end - start

        if multi:
            time_multi.append(diff)

        else:
            time_single.append(diff)

    speedup.append(time_single[-1] / time_multi[-1])
    print(n_spheres, "", time_single[-1], time_multi[-1], speedup[-1], sep="\t")

print("mean", "", np.mean(time_single), np.mean(time_multi), np.mean(speedup), sep="\t")

# plt.show()
