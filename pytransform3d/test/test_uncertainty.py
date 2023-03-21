import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu
from numpy.testing import assert_array_almost_equal
import pytest


def test_same_fuse_poses():
    mean1 = np.array([
        [0.8573, -0.2854, 0.4285, 3.5368],
        [-0.1113, 0.7098, 0.6956, -3.5165],
        [-0.5026, -0.6440, 0.5767, -0.9112],
        [0.0, 0.0, 0.0, 1.0000]
    ])
    mean1[:3, :3] = pr.norm_matrix(mean1[:3, :3])
    mean2 = np.array([
        [0.5441, -0.6105, 0.5755, -1.0935],
        [0.8276, 0.5032, -0.2487, 5.5992],
        [-0.1377, 0.6116, 0.7791, 0.2690],
        [0.0, 0.0, 0.0, 1.0000]
    ])
    mean2[:3, :3] = pr.norm_matrix(mean2[:3, :3])
    mean3 = np.array([
        [-0.0211, -0.7869, 0.6167, -3.0968],
        [-0.2293, 0.6042, 0.7631, 2.0868],
        [-0.9731, -0.1254, -0.1932, 2.0239],
        [0.0, 0.0, 0.0, 1.0000]
    ])
    mean3[:3, :3] = pr.norm_matrix(mean3[:3, :3])
    alpha = 5.0
    cov1 = alpha * np.diag([0.1, 0.2, 0.1, 2.0, 1.0, 1.0])
    cov2 = alpha * np.diag([0.1, 0.1, 0.2, 1.0, 3.0, 1.0])
    cov3 = alpha * np.diag([0.2, 0.1, 0.1, 1.0, 1.0, 5.0])
    means = [mean1, mean2, mean3]
    covs = [cov1, cov2, cov3]
    mean_est, cov_est, V = pu.pose_fusion(means, covs)
    mean_exp = np.array([
        [0.2967, -0.7157, 0.6323, -1.4887],
        [0.5338, 0.6733, 0.5116, 0.9935],
        [-0.7918, 0.1857, 0.5818, -2.7035],
        [0.0, 0.0, 0.0, 1.0000]
    ])
    cov_exp = np.array([
        [0.14907707, -0.01935277, -0.0107348, -0.02442925, -0.09843835, 0.0054134],
        [-0.01935277, 0.14648459, 0.02055571, 0.11121064, 0.06272014, -0.08553834],
        [-0.0107348, 0.02055571, 0.15260209, -0.07451066, 0.06531188, -0.01890897],
        [-0.02442925, 0.11121064, -0.07451066, 2.10256906, 0.13695598, -0.29705468],
        [-0.09843835, 0.06272014, 0.06531188, 0.13695598, 2.29286157, -0.58004],
        [0.0054134, -0.08553834, -0.01890897, -0.29705468, -0.58004, 2.34528443]])
    assert_array_almost_equal(mean_exp, mean_est, decimal=4)
    assert_array_almost_equal(cov_exp, cov_est)
    assert pytest.approx(V, abs=1e-4) == 4.6537


def test_invert_pose():
    rng = np.random.default_rng(2)

    for _ in range(5):
        T = pt.random_transform(rng)
        cov = np.diag(rng.random((6, 6)))

        T_inv, cov_inv = pu.invert_uncertain_transform(T, cov)

        T2, cov2 = pu.invert_uncertain_transform(T_inv, cov_inv)

        assert_array_almost_equal(T, T2)
        assert_array_almost_equal(cov, cov2)


def test_sample_estimate_gaussian():
    rng = np.random.default_rng(2000)
    mean = pt.transform_from(R=np.eye(3), p=np.array([0.0, 0.0, 0.5]))
    cov = np.diag([0.001, 0.001, 0.5, 0.001, 0.001, 0.001])
    samples = np.array([pt.random_transform(rng, mean, cov)
                        for _ in range(1000)])
    mean_est, cov_est = pu.estimate_gaussian_transform_from_samples(samples)
    assert_array_almost_equal(mean, mean_est, decimal=2)
    assert_array_almost_equal(cov, cov_est, decimal=2)
