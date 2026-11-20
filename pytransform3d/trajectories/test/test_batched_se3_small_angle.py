import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.linalg import expm

from pytransform3d.trajectories import transforms_from_exponential_coordinates


@pytest.mark.parametrize("angle", [1e-12, 1e-8, 1e-6, 1e-4, 0.1, 1.0])
def test_batched_exponential_keeps_small_rotation_translation_coupling(angle):
    twists = np.array(
        [[0.0, 0.0, angle, 1.0, 2.0, 3.0], [angle, 0.0, 0.0, -2.0, 1.0, 0.5]]
    )
    expected = []
    for wx, wy, wz, vx, vy, vz in twists:
        algebra = np.array(
            [[0, -wz, wy, vx], [wz, 0, -wx, vy], [-wy, wx, 0, vz], [0, 0, 0, 0]]
        )
        expected.append(expm(algebra))
    assert_allclose(
        transforms_from_exponential_coordinates(twists),
        expected,
        rtol=0.0,
        atol=2e-14,
    )


def test_batched_exponential_handles_mixed_zero_and_small_rotations():
    twists = np.zeros((2, 3, 6))
    twists[..., 3:] = [1.0, 2.0, 3.0]
    twists[0, 1, 2] = 1e-8
    result = transforms_from_exponential_coordinates(twists)
    assert_allclose(result[0, 0, :3, 3], [1, 2, 3])
    assert_allclose(
        result[0, 1, :3, 3], [1 - 1e-8, 2 + 0.5e-8, 3], rtol=0, atol=2e-14
    )
