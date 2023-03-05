import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu
from numpy.testing import assert_array_almost_equal
import pytest


def test_jacobian_so3():
    rng = np.random.default_rng(0)
    omega = pr.random_compact_axis_angle(rng)
    J = pu.left_jacobian_SO3(omega)
    J_inv = pu.left_jacobian_SO3_inv(omega)
    J_inv_J = np.dot(J_inv, J)
    assert_array_almost_equal(J_inv_J, np.eye(3))


def test_jacobian_se3():
    rng = np.random.default_rng(0)
    Stheta = pt.random_exponential_coordinates(rng)
    J = pu.jacobian_SE3(Stheta)
    J_inv = pu.jacobian_SE3_inv(Stheta)
    J_inv_J = np.dot(J_inv, J)
    assert_array_almost_equal(J_inv_J, np.eye(6))


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
    #for c in means:
    #    print(f"""
    #[
    #       {' '.join(map(str, c[0]))};
    #       {' '.join(map(str, c[1]))};
    #       {' '.join(map(str, c[2]))};
    #       {' '.join(map(str, c[3]))};
    #   ]
    #    """)
    mean_est, cov_est, V = pu.fuse_poses(means, covs, return_error=True)
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
