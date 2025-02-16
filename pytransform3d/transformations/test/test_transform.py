import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def test_check_transform():
    """Test input validation for transformation matrix."""
    A2B = np.eye(3)
    with pytest.raises(
        ValueError,
        match="Expected homogeneous transformation matrix with shape",
    ):
        pt.check_transform(A2B)

    A2B = np.eye(4, dtype=int)
    A2B = pt.check_transform(A2B)
    assert type(A2B) is np.ndarray
    assert A2B.dtype == np.float64

    A2B[:3, :3] = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]])
    with pytest.raises(ValueError, match="rotation matrix"):
        pt.check_transform(A2B)

    A2B = np.eye(4)
    A2B[3, :] = np.array([0.1, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="homogeneous transformation matrix"):
        pt.check_transform(A2B)

    rng = np.random.default_rng(0)
    A2B = pt.random_transform(rng)
    A2B2 = pt.check_transform(A2B)
    assert_array_almost_equal(A2B, A2B2)


def test_deactivate_transform_precision_error():
    A2B = np.eye(4)
    A2B[0, 0] = 2.0
    A2B[3, 0] = 3.0
    with pytest.raises(ValueError, match="Expected rotation matrix"):
        pt.check_transform(A2B)

    n_expected_warnings = 2
    try:
        warnings.filterwarnings("always", category=UserWarning)
        with warnings.catch_warnings(record=True) as w:
            pt.check_transform(A2B, strict_check=False)
            assert len(w) == n_expected_warnings
    finally:
        warnings.filterwarnings("default", category=UserWarning)


def test_transform_requires_renormalization():
    assert pt.transform_requires_renormalization(np.eye(4) + 1e-6)
    assert not pt.transform_requires_renormalization(np.eye(4))


def test_translate_transform_with_check():
    A2B_broken = np.zeros((4, 4))
    with pytest.raises(ValueError, match="rotation matrix"):
        pt.translate_transform(A2B_broken, np.zeros(3))


def test_rotate_transform_with_check():
    A2B_broken = np.zeros((4, 4))
    with pytest.raises(ValueError, match="rotation matrix"):
        pt.rotate_transform(A2B_broken, np.eye(3))


def test_pq_from_transform():
    """Test conversion from homogeneous matrix to position and quaternion."""
    A2B = np.eye(4)
    pq = pt.pq_from_transform(A2B)
    assert_array_almost_equal(pq, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))


def test_exponential_coordinates_from_almost_identity_transform():
    A2B = np.array(
        [
            [
                0.9999999999999999,
                -1.5883146449068575e-16,
                4.8699079321578667e-17,
                -7.54265065748827e-05,
            ],
            [
                5.110044286978025e-17,
                0.9999999999999999,
                1.1798895336935056e-17,
                9.340523179823812e-05,
            ],
            [
                3.0048299647976294e-18,
                5.4741890703482423e-17,
                1.0,
                -7.803584869947588e-05,
            ],
            [0, 0, 0, 1],
        ]
    )
    Stheta = pt.exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(np.zeros(6), Stheta, decimal=4)


def test_exponential_coordinates_from_transform_without_check():
    transform = np.ones((4, 4))
    Stheta = pt.exponential_coordinates_from_transform(transform, check=False)
    assert_array_almost_equal(Stheta, np.array([0, 0, 0, 1, 1, 1]))


def test_transform_log_from_almost_identity_transform():
    A2B = np.array(
        [
            [
                0.9999999999999999,
                -1.5883146449068575e-16,
                4.8699079321578667e-17,
                -7.54265065748827e-05,
            ],
            [
                5.110044286978025e-17,
                0.9999999999999999,
                1.1798895336935056e-17,
                9.340523179823812e-05,
            ],
            [
                3.0048299647976294e-18,
                5.4741890703482423e-17,
                1.0,
                -7.803584869947588e-05,
            ],
            [0, 0, 0, 1],
        ]
    )
    transform_log = pt.transform_log_from_transform(A2B)
    assert_array_almost_equal(np.zeros((4, 4)), transform_log)


def test_conversions_between_dual_quaternion_and_transform():
    rng = np.random.default_rng(1000)
    for _ in range(5):
        A2B = pt.random_transform(rng)
        dq = pt.dual_quaternion_from_transform(A2B)
        A2B2 = pt.transform_from_dual_quaternion(dq)
        assert_array_almost_equal(A2B, A2B2)
        dq2 = pt.dual_quaternion_from_transform(A2B2)
        pt.assert_unit_dual_quaternion_equal(dq, dq2)
    for _ in range(5):
        p = pr.random_vector(rng, 3)
        q = pr.random_quaternion(rng)
        dq = pt.dual_quaternion_from_pq(np.hstack((p, q)))
        A2B = pt.transform_from_dual_quaternion(dq)
        dq2 = pt.dual_quaternion_from_transform(A2B)
        pt.assert_unit_dual_quaternion_equal(dq, dq2)
        A2B2 = pt.transform_from_dual_quaternion(dq2)
        assert_array_almost_equal(A2B, A2B2)


def test_adjoint_of_transformation():
    rng = np.random.default_rng(94)
    for _ in range(5):
        A2B = pt.random_transform(rng)
        theta_dot = 3.0 * float(rng.random())
        S = pt.random_screw_axis(rng)

        V_A = S * theta_dot

        adj_A2B = pt.adjoint_from_transform(A2B)
        V_B = adj_A2B.dot(V_A)

        S_mat = pt.screw_matrix_from_screw_axis(S)
        V_mat_A = S_mat * theta_dot
        V_mat_B = np.dot(np.dot(A2B, V_mat_A), pt.invert_transform(A2B))

        S_B, theta_dot2 = pt.screw_axis_from_exponential_coordinates(V_B)
        V_mat_B2 = pt.screw_matrix_from_screw_axis(S_B) * theta_dot2
        assert pytest.approx(theta_dot) == theta_dot2
        assert_array_almost_equal(V_mat_B, V_mat_B2)


def test_adjoint_from_transform_without_check():
    transform = np.ones((4, 4))
    adjoint = pt.adjoint_from_transform(transform, check=False)
    assert_array_almost_equal(adjoint[:3, :3], np.ones((3, 3)))
    assert_array_almost_equal(adjoint[3:, 3:], np.ones((3, 3)))
    assert_array_almost_equal(adjoint[3:, :3], np.zeros((3, 3)))
    assert_array_almost_equal(adjoint[:3, 3:], np.zeros((3, 3)))
