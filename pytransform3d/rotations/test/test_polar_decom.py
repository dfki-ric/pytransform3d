import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_polar_decomposition():
    # no iterations required
    R_norm = pr.robust_polar_decomposition(np.eye(3), n_iter=1, eps=1e-32)
    assert_array_almost_equal(R_norm, np.eye(3))

    rng = np.random.default_rng(33)

    # scaled valid rotation matrix
    R = pr.random_matrix(rng)
    R_norm = pr.robust_polar_decomposition(23.5 * R)
    assert_array_almost_equal(R_norm, R)

    # slightly rotated basis vector
    R = pr.matrix_from_euler(np.deg2rad([-45, 45, 45]), 2, 1, 0, True)
    R_unnormalized = np.copy(R)
    R_unnormalized[:, 0] = np.dot(
        pr.active_matrix_from_angle(2, np.deg2rad(1.0)), R_unnormalized[:, 0]
    )
    # norm_matrix will just fix the orthogonality with the cross product of the
    # two other basis vectors
    assert_array_almost_equal(pr.norm_matrix(R_unnormalized), R)
    # polar decomposition will spread the error more evenly among basis vectors
    errors = np.linalg.norm((R_unnormalized - R).T, axis=-1)
    assert pytest.approx(errors.tolist()) == [0.00909314, 0, 0]
    R_norm = pr.robust_polar_decomposition(R_unnormalized)
    norm_errors = np.linalg.norm((R_norm - R).T, axis=-1)
    assert np.all(norm_errors > 0)
    assert np.all(norm_errors < errors[0])
    assert np.std(norm_errors) < np.std(errors)

    # random rotations of random basis vectors
    for _ in range(5):
        R = pr.random_matrix(rng)
        R_unnormalized = np.copy(R)
        random_axis = rng.integers(0, 3)
        R_unnormalized[:, random_axis] = np.dot(
            pr.active_matrix_from_angle(
                rng.integers(0, 3), np.deg2rad(rng.uniform(-3, 3))
            ),
            R_unnormalized[:, random_axis],
        )
        R_norm = pr.robust_polar_decomposition(R_unnormalized)
        errors = np.linalg.norm((R_unnormalized - R).T, axis=-1)
        norm_errors = np.linalg.norm((R_norm - R).T, axis=-1)
        assert np.all(norm_errors > 0)
        assert np.all(norm_errors < errors.sum())
        assert np.std(norm_errors) < np.std(errors)
