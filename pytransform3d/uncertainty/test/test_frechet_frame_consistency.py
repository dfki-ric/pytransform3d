import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import logm
from scipy.spatial.transform import Rotation

import pytransform3d.batch_rotations as pbr
import pytransform3d.trajectories as ptr
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu


def compact_axis_angles(matrices):
    axes = pbr.axis_angles_from_matrices(matrices)
    return axes[..., :3] * axes[..., 3, np.newaxis]


def test_one_update_returns_centered_translation_residuals():
    samples = np.array([[0.2, 0.1, -0.3], [0.8, -0.2, 0.5], [-0.1, 0.4, 0.7]])
    mean0 = np.array([0.4, -0.3, 0.2])
    saved = samples.copy()
    mean, residuals = pu.frechet_mean(
        samples,
        mean0,
        exp=lambda v: v,
        log=lambda v: v,
        inv=lambda v: -v,
        concat_one_to_one=lambda a, b: a + b,
        concat_many_to_one=lambda a, b: a + b,
        n_iter=1,
    )
    expected_mean = samples.mean(axis=0)
    assert_allclose(mean, expected_mean, rtol=0, atol=1e-15)
    assert_allclose(residuals, samples - expected_mean, rtol=0, atol=1e-15)
    assert_allclose(residuals.mean(axis=0), np.zeros(3), rtol=0, atol=1e-15)
    assert_array_equal(samples, saved)


@pytest.mark.parametrize("n_iter", [1, 2, 4])
def test_noncommuting_rotation_residuals_match_returned_mean(n_iter):
    samples = Rotation.from_rotvec(
        [[0.8, 0.2, -0.1], [-0.1, 0.7, 0.3], [0.2, -0.4, 0.6]]
    ).as_matrix()
    mean0 = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    mean, residuals = pu.frechet_mean(
        samples,
        mean0,
        exp=pbr.matrices_from_compact_axis_angles,
        log=compact_axis_angles,
        inv=lambda matrix: matrix.T,
        concat_one_to_one=lambda a, b: b @ a,
        concat_many_to_one=ptr.concat_many_to_one,
        n_iter=n_iter,
    )
    # SciPy independently computes Log(returned_mean.T @ R_i).
    expected = Rotation.from_matrix(mean.T @ samples).as_rotvec()
    assert_allclose(residuals, expected, rtol=1e-11, atol=1e-12)
    assert_allclose(
        np.cov(residuals, rowvar=False),
        np.cov(expected, rowvar=False),
        rtol=1e-11,
        atol=1e-12,
    )


@pytest.mark.parametrize("n_iter", [1, 2, 4])
def test_noncommuting_pose_residuals_match_matrix_logarithm(n_iter):
    rotations = Rotation.from_rotvec(
        [[0.8, 0.2, -0.1], [-0.1, 0.7, 0.3], [0.2, -0.4, 0.6]]
    ).as_matrix()
    samples = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    samples[:, :3, :3] = rotations
    samples[:, :3, 3] = [[0.2, -0.3, 0.1], [-0.1, 0.4, 0.5], [0.3, 0.1, -0.2]]
    mean0 = np.eye(4)
    mean0[:3, :3] = Rotation.from_rotvec([0.3, -0.2, 0.1]).as_matrix()
    mean0[:3, 3] = [0.1, -0.2, 0.3]
    mean, residuals = pu.frechet_mean(
        samples,
        mean0,
        exp=ptr.transforms_from_exponential_coordinates,
        log=ptr.exponential_coordinates_from_transforms,
        inv=pt.invert_transform,
        concat_one_to_one=pt.concat,
        concat_many_to_one=ptr.concat_many_to_one,
        n_iter=n_iter,
    )
    expected = []
    inverse_mean = np.linalg.inv(mean)
    for sample in samples:
        generator = logm(inverse_mean @ sample)
        assert_allclose(generator.imag, 0, rtol=0, atol=1e-14)
        generator = generator.real
        expected.append(
            [
                generator[2, 1],
                generator[0, 2],
                generator[1, 0],
                generator[0, 3],
                generator[1, 3],
                generator[2, 3],
            ]
        )
    expected = np.asarray(expected)
    assert_allclose(residuals, expected, rtol=1e-10, atol=1e-12)
    assert_allclose(
        np.cov(residuals, rowvar=False),
        np.cov(expected, rowvar=False),
        rtol=1e-10,
        atol=1e-12,
    )


def symmetric_global_rotations(mean, amplitudes):
    vectors = np.concatenate((np.diag(amplitudes), -np.diag(amplitudes)))
    return Rotation.from_rotvec(vectors).as_matrix() @ mean


@pytest.mark.parametrize("angle", [0.5 * np.pi, 0.8 * np.pi])
def test_gaussian_rotation_mean_and_covariance_use_global_frame(angle):
    mean = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
    amplitudes = np.array([0.1, 0.2, 0.3])
    samples = symmetric_global_rotations(mean, amplitudes)
    actual_mean, actual_cov = pu.estimate_gaussian_rotation_matrix_from_samples(
        samples
    )
    # Each independent +/- pair has opposite global logarithms at mean.
    # All samples lie in its unique small convex neighbourhood.
    expected_cov = (2.0 / (len(samples) - 1)) * np.diag(amplitudes**2)
    assert_allclose(actual_mean, mean, rtol=0, atol=1e-11)
    assert_allclose(actual_cov, expected_cov, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("angle", [0.5 * np.pi, 0.8 * np.pi])
def test_gaussian_pose_translation_mean_and_covariance_use_global_frame(angle):
    mean = np.eye(4)
    mean[:3, :3] = Rotation.from_rotvec([0.0, 0.0, angle]).as_matrix()
    mean[:3, 3] = [0.2, -0.4, 0.3]
    amplitudes = np.array([0.1, 0.2, 0.3])
    translations = np.concatenate((np.diag(amplitudes), -np.diag(amplitudes)))
    samples = np.broadcast_to(mean, (len(translations), 4, 4)).copy()
    samples[:, :3, 3] += translations
    actual_mean, actual_cov = pu.estimate_gaussian_transform_from_samples(
        samples
    )
    expected_cov = np.zeros((6, 6))
    expected_cov[3:, 3:] = (2.0 / (len(samples) - 1)) * np.diag(amplitudes**2)
    assert_allclose(actual_mean, mean, rtol=0, atol=1e-11)
    assert_allclose(actual_cov, expected_cov, rtol=1e-10, atol=1e-12)
