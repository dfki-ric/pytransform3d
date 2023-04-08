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
        [0.14907707, -0.01935277, -0.0107348,
         -0.02442925, -0.09843835, 0.0054134],
        [-0.01935277, 0.14648459, 0.02055571,
         0.11121064, 0.06272014, -0.08553834],
        [-0.0107348, 0.02055571, 0.15260209,
         -0.07451066, 0.06531188, -0.01890897],
        [-0.02442925, 0.11121064, -0.07451066,
         2.10256906, 0.13695598, -0.29705468],
        [-0.09843835, 0.06272014, 0.06531188,
         0.13695598, 2.29286157, -0.58004],
        [0.0054134, -0.08553834, -0.01890897,
         -0.29705468, -0.58004, 2.34528443]])
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


def test_concat_globally_uncertain_transforms():
    cov_pose_chol = np.diag([0, 0, 0.03, 0, 0, 0])
    cov_pose = np.dot(cov_pose_chol, cov_pose_chol.T)
    velocity_vector = np.array([0, 0, 0, 1.0, 0, 0])
    T_vel = pt.transform_from_exponential_coordinates(velocity_vector)
    n_steps = 100

    T_est = np.eye(4)
    cov_est = np.zeros((6, 6))
    for t in range(n_steps):
        T_est, cov_est = pu.concat_globally_uncertain_transforms(
            T_est, cov_est, T_vel, cov_pose)
    assert_array_almost_equal(
        T_est, np.array([
            [1, 0, 0, n_steps],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))
    # achievable with second-order terms
    assert cov_est[2, 2] > 0
    assert cov_est[4, 4] > 0
    assert cov_est[2, 4] != 0
    assert cov_est[4, 2] != 0
    # achievable only with fourth-order terms
    assert cov_est[3, 3] > 0
    assert cov_est[4, 2] != 0
    assert cov_est[2, 4] != 0


def test_concat_locally_uncertain_transforms():
    mean_A2B = pt.transform_from(
        R=pr.matrix_from_euler([0.5, 0.0, 0.0], 0, 1, 2, True),
        p=np.array([1.0, 0.0, 0.0])
    )
    mean_B2C = pt.transform_from(
        R=pr.matrix_from_euler([0.0, 0.5, 0.0], 0, 1, 2, True),
        p=np.array([0.0, 1.0, 0.0])
    )
    cov_A = np.diag([1.0, 0, 0, 0, 0, 0])
    cov_B = np.diag([0, 1.0, 0, 0, 0, 0])

    mean_A2C, cov_A_total = pu.concat_locally_uncertain_transforms(
        mean_A2B, mean_B2C, cov_A, np.zeros((6, 6)))
    assert_array_almost_equal(mean_A2C, pt.concat(mean_A2B, mean_B2C))
    assert_array_almost_equal(cov_A_total, cov_A)

    mean_A2C, cov_A_total = pu.concat_locally_uncertain_transforms(
        np.eye(4), mean_B2C, np.zeros((6, 6)), cov_B)
    assert_array_almost_equal(cov_A_total, cov_B)

    mean_A2C, cov_A_total = pu.concat_locally_uncertain_transforms(
        mean_A2B, mean_B2C, np.zeros((6, 6)), cov_B)
    ad_B2A = pt.adjoint_from_transform(pt.invert_transform(mean_A2B))
    assert_array_almost_equal(cov_A_total, ad_B2A.dot(cov_B).dot(ad_B2A.T))


def test_to_ellipsoid():
    mean = np.array([0.1, 0.2, 0.3])
    cov = np.array([
        [25.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 9.0]
    ])
    ellipsoid2origin, radii = pu.to_ellipsoid(mean, cov)
    assert_array_almost_equal(ellipsoid2origin[:3, 3], mean)
    assert_array_almost_equal(
        ellipsoid2origin[:3, :3],
        np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])
    )
    assert_array_almost_equal(radii, np.array([2.0, 3.0, 5.0]))

    rng = np.random.default_rng(28)
    for _ in range(5):
        R = pr.matrix_from_axis_angle(pr.random_axis_angle(rng))
        std_devs = np.array([1.0, 2.0, 3.0])
        cov = np.dot(R, np.dot(np.diag(std_devs ** 2), R.T))
        ellipsoid2origin, radii = pu.to_ellipsoid(mean, cov)
        assert_array_almost_equal(ellipsoid2origin[:3, 3], mean)
        # multiple solutions for the rotation matrix are possible because of
        # ellipsoid symmetries, hence, we only check the absolute values
        assert_array_almost_equal(np.abs(ellipsoid2origin[:3, :3]), np.abs(R))
        assert_array_almost_equal(radii, std_devs)


def test_projected_ellipsoid():
    mean = np.eye(4)
    cov = np.eye(6)
    x, y, z = pu.to_projected_ellipsoid(mean, cov, factor=1.0, n_steps=20)
    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    r = np.linalg.norm(P, axis=1)
    assert_array_almost_equal(r, np.ones_like(3))

    pos_mean = np.array([0.5, -5.3, 10.5])
    mean = pt.transform_from(R=np.eye(3), p=pos_mean)
    x, y, z = pu.to_projected_ellipsoid(mean, cov, factor=1.0, n_steps=20)
    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    r = np.linalg.norm(P - pos_mean[np.newaxis], axis=1)
    assert_array_almost_equal(r, np.ones_like(3))

    mean = np.eye(4)
    cov = np.diag([1, 1, 1, 4, 1, 1])
    x, y, z = pu.to_projected_ellipsoid(mean, cov, factor=1.0, n_steps=20)
    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    r = np.linalg.norm(P, axis=1)
    assert np.all(1.0 <= r)
    assert np.all(r <= 2.0)
