"""Integer angle arrays must allow fractional screw translations."""

import numpy as np
import pytest
from pytransform3d.trajectories import dual_quaternions_from_screw_parameters


@pytest.mark.parametrize("shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize(
    "dtype", [np.int32, np.int64, np.uint32, np.float32, np.float64]
)
@pytest.mark.parametrize("pitch", [0.0, 0.5, -1.25, np.inf])
def test_integer_angles_fractional_pitch(shape, dtype, pitch):
    angles = np.full(shape, 1, dtype=dtype)
    pitches = np.full(shape, pitch)
    points = np.zeros(shape + (3,))
    axes = np.zeros(shape + (3,))
    axes[..., 2] = 1.0
    before = [x.copy() for x in (points, axes, pitches, angles)]
    actual = dual_quaternions_from_screw_parameters(
        points, axes, pitches, angles
    )
    # Analytic z rotation, axis through origin: d = pitch * theta.
    expected = np.zeros(shape + (8,))
    half_angle = 0.0 if np.isinf(pitch) else 0.5
    distance = 1.0 if np.isinf(pitch) else pitch
    expected[..., 0] = np.cos(half_angle)
    expected[..., 3] = np.sin(half_angle)
    expected[..., 4] = -0.5 * distance * np.sin(half_angle)
    expected[..., 7] = 0.5 * distance * np.cos(half_angle)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    for value, original in zip((points, axes, pitches, angles), before):
        np.testing.assert_array_equal(value, original)


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_empty_integer_angles(shape):
    result = dual_quaternions_from_screw_parameters(
        np.empty(shape + (3,)),
        np.empty(shape + (3,)),
        np.empty(shape),
        np.empty(shape, dtype=int),
    )
    assert result.shape == shape + (8,)


def test_mixed_pitches_readonly_integer_angles():
    angles = np.array([0, 1, 2, 3], dtype=np.int64)
    pitches = np.array([0.5, np.inf, -0.25, 0.0])
    angles.setflags(write=False)
    pitches.setflags(write=False)
    points = np.zeros((4, 3))
    axes = np.tile([0.0, 0.0, 1.0], (4, 1))
    result = dual_quaternions_from_screw_parameters(
        points, axes, pitches, angles
    )
    rotation_angles = np.array([0.0, 0.0, 2.0, 3.0])
    distances = np.array([0.0, 1.0, -0.5, 0.0])
    expected = np.zeros((4, 8))
    expected[:, 0] = np.cos(0.5 * rotation_angles)
    expected[:, 3] = np.sin(0.5 * rotation_angles)
    expected[:, 4] = -0.5 * distances * np.sin(0.5 * rotation_angles)
    expected[:, 7] = 0.5 * distances * np.cos(0.5 * rotation_angles)
    np.testing.assert_allclose(result, expected, atol=1e-15)
    np.testing.assert_array_equal(angles, [0, 1, 2, 3])
    np.testing.assert_array_equal(pitches, [0.5, np.inf, -0.25, 0.0])
