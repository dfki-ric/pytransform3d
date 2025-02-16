import numpy as np

import pytransform3d.transformations as pt


def test_random_screw_axis():
    rng = np.random.default_rng(893)
    for _ in range(5):
        S = pt.random_screw_axis(rng)
        pt.check_screw_axis(S)
