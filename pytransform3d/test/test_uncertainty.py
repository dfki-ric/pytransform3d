import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.uncertainty as pu
from numpy.testing import assert_array_almost_equal


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
