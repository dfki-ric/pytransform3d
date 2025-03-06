import numpy as np
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_random_matrices():
    rng = np.random.default_rng(33)
    mean = pr.random_matrix(rng, np.eye(3), np.eye(3))
    cov = np.eye(3) * 1e-10
    samples = np.stack([pr.random_matrix(rng, mean, cov) for _ in range(10)])
    for sample in samples:
        assert_array_almost_equal(sample, mean, decimal=4)
