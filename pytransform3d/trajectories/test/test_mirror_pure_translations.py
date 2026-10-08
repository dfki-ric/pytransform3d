import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm

import pytransform3d.trajectories as ptr


def se3_generator_matrix(coordinates):
    wx, wy, wz, vx, vy, vz = coordinates
    return np.array(
        [[0, -wz, wy, vx], [wz, 0, -wx, vy], [-wy, wx, 0, vz], [0, 0, 0, 0]],
        dtype=float,
    )


def test_mirror_preserves_straight_translation_trajectory():
    coordinates = np.zeros((5, 6))
    coordinates[:, 3:] = np.arange(5)[:, None] * np.array([0.2, -0.1, 0.3])
    saved = coordinates.copy()
    mirrored = ptr.mirror_screw_axis_direction(coordinates)
    assert_array_equal(mirrored, coordinates)
    assert_array_equal(coordinates, saved)
    for original, transformed in zip(coordinates, mirrored):
        assert_allclose(
            expm(se3_generator_matrix(transformed)),
            expm(se3_generator_matrix(original)),
            rtol=1e-14,
            atol=1e-14,
        )


def test_mixed_stationary_translation_and_rotation_segments():
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2, -0.3, 0.1],
            [0.1, -0.2, 0.3, 0.4, -0.2, 0.5],
            [0.0, 0.0, 0.0, -0.5, 0.2, 0.7],
        ]
    )
    with np.errstate(all="raise"):
        mirrored = ptr.mirror_screw_axis_direction(coordinates)
    assert np.isfinite(mirrored).all()
    for original, transformed in zip(coordinates, mirrored):
        assert_allclose(
            expm(se3_generator_matrix(transformed)),
            expm(se3_generator_matrix(original)),
            rtol=1e-13,
            atol=1e-13,
        )
    assert_array_equal(mirrored[[0, 1, 3]], coordinates[[0, 1, 3]])


def test_empty_and_readonly_translation_inputs():
    assert ptr.mirror_screw_axis_direction(np.empty((0, 6))).shape == (0, 6)
    coordinates = np.array([[0.0, 0.0, 0.0, 0.2, -0.1, 0.3]])
    coordinates.flags.writeable = False
    actual = ptr.mirror_screw_axis_direction(coordinates)
    assert_array_equal(actual, coordinates)
    actual[0, 3] = 0.8
    assert coordinates[0, 3] == 0.2


def test_mirror_preserves_rotations_below_screw_axis_threshold():
    eps = np.finfo(float).eps
    rotations = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5 * eps, 0.0, 0.0],
            [-0.5 * eps, 0.0, 0.0],
            [0.0, np.nextafter(eps, 0.0), 0.0],
            [0.5 * eps, -0.5 * eps, 0.5 * eps],
        ]
    )
    coordinates = np.column_stack(
        (rotations, np.tile([0.2, -0.3, 0.1], (len(rotations), 1)))
    )
    saved = coordinates.copy()
    with np.errstate(all="raise"):
        mirrored = ptr.mirror_screw_axis_direction(coordinates)
    assert_array_equal(mirrored, coordinates)
    assert_array_equal(coordinates, saved)
    for original, transformed in zip(coordinates, mirrored):
        assert_allclose(
            expm(se3_generator_matrix(transformed)),
            expm(se3_generator_matrix(original)),
        )


def test_mirror_rotates_at_and_above_screw_axis_threshold():
    eps = np.finfo(float).eps
    rotations = np.array(
        [[eps, 0.0, 0.0], [0.0, -2.0 * eps, 0.0], [0.75 * eps] * 3]
    )
    coordinates = np.column_stack((rotations, np.zeros_like(rotations)))
    with np.errstate(all="raise"):
        mirrored = ptr.mirror_screw_axis_direction(coordinates)
    angles = np.linalg.norm(rotations, axis=1)
    assert_allclose(np.linalg.norm(mirrored[:, :3], axis=1), 2 * np.pi - angles)
    for original, transformed in zip(coordinates, mirrored):
        assert_allclose(
            expm(se3_generator_matrix(transformed)),
            expm(se3_generator_matrix(original)),
            rtol=1e-13,
            atol=1e-13,
        )
