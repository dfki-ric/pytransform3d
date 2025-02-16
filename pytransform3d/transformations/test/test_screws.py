import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def test_check_screw_parameters():
    q = [100.0, -20.3, 1e-3]
    s_axis = pr.norm_vector(np.array([-1.0, 2.0, 3.0])).tolist()
    h = 0.2

    with pytest.raises(ValueError, match="Expected 3D vector with shape"):
        pt.check_screw_parameters([0.0], s_axis, h)
    with pytest.raises(ValueError, match="Expected 3D vector with shape"):
        pt.check_screw_parameters(q, [0.0], h)
    with pytest.raises(ValueError, match="s_axis must not have norm 0"):
        pt.check_screw_parameters(q, np.zeros(3), h)

    q2, s_axis2, h2 = pt.check_screw_parameters(q, 2.0 * np.array(s_axis), h)
    assert_array_almost_equal(q, q2)
    assert pytest.approx(h) == h2
    assert pytest.approx(np.linalg.norm(s_axis2)) == 1.0

    q2, s_axis2, h2 = pt.check_screw_parameters(q, s_axis, h)
    assert_array_almost_equal(q, q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert pytest.approx(h) == h2

    q2, s_axis2, h2 = pt.check_screw_parameters(q, s_axis, np.inf)
    assert_array_almost_equal(np.zeros(3), q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert np.isinf(h2)


def test_check_screw_axis():
    rng = np.random.default_rng(73)
    omega = pr.norm_vector(pr.random_vector(rng, 3))
    v = pr.random_vector(rng, 3)

    with pytest.raises(ValueError, match="Expected 3D vector with shape"):
        pt.check_screw_axis(np.r_[0, 1, v])

    with pytest.raises(
        ValueError, match="Norm of rotation axis must either be 0 or 1"
    ):
        pt.check_screw_axis(np.r_[2 * omega, v])

    with pytest.raises(
        ValueError, match="If the norm of the rotation axis is 0"
    ):
        pt.check_screw_axis(np.r_[0, 0, 0, v])

    S_pure_translation = np.r_[0, 0, 0, pr.norm_vector(v)]
    S = pt.check_screw_axis(S_pure_translation)
    assert_array_almost_equal(S, S_pure_translation)

    S_both = np.r_[omega, v]
    S = pt.check_screw_axis(S_both)
    assert_array_almost_equal(S, S_both)


def test_check_exponential_coordinates():
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_exponential_coordinates([0])

    Stheta = [0.0, 1.0, 2.0, -5.0, -2, 3]
    Stheta2 = pt.check_exponential_coordinates(Stheta)
    assert_array_almost_equal(Stheta, Stheta2)


def test_check_screw_matrix():
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_screw_matrix(np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_screw_matrix(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_screw_matrix(np.zeros((4, 3)))

    with pytest.raises(
        ValueError, match="Last row of screw matrix must only " "contains zeros"
    ):
        pt.check_screw_matrix(np.eye(4))

    screw_matrix = (
        pt.screw_matrix_from_screw_axis(
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
        * 1.1
    )
    with pytest.raises(
        ValueError, match="Norm of rotation axis must either be 0 or 1"
    ):
        pt.check_screw_matrix(screw_matrix)

    screw_matrix = (
        pt.screw_matrix_from_screw_axis(
            np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
        * 1.1
    )
    with pytest.raises(
        ValueError, match="If the norm of the rotation axis is 0"
    ):
        pt.check_screw_matrix(screw_matrix)

    screw_matrix = pt.screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    )
    screw_matrix2 = pt.check_screw_matrix(screw_matrix)
    assert_array_almost_equal(screw_matrix, screw_matrix2)

    screw_matrix = pt.screw_matrix_from_screw_axis(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    )
    screw_matrix2 = pt.check_screw_matrix(screw_matrix)
    assert_array_almost_equal(screw_matrix, screw_matrix2)

    screw_matrix = pt.screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    )
    screw_matrix[0, 0] = 0.0001
    with pytest.raises(ValueError, match="Expected skew-symmetric matrix"):
        pt.check_screw_matrix(screw_matrix)


def test_check_transform_log():
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_transform_log(np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_transform_log(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        pt.check_transform_log(np.zeros((4, 3)))

    with pytest.raises(
        ValueError,
        match="Last row of logarithm of transformation must "
        "only contains zeros",
    ):
        pt.check_transform_log(np.eye(4))
    transform_log = (
        pt.screw_matrix_from_screw_axis(
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
        * 1.1
    )
    transform_log2 = pt.check_transform_log(transform_log)
    assert_array_almost_equal(transform_log, transform_log2)

    transform_log = (
        pt.screw_matrix_from_screw_axis(
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        )
        * 1.1
    )
    transform_log[0, 0] = 0.0001
    with pytest.raises(ValueError, match="Expected skew-symmetric matrix"):
        pt.check_transform_log(transform_log)


def test_norm_exponential_coordinates():
    Stheta_only_translation = np.array([0.0, 0.0, 0.0, 100.0, 25.0, -23.0])
    Stheta_only_translation2 = pt.norm_exponential_coordinates(
        Stheta_only_translation
    )
    assert_array_almost_equal(Stheta_only_translation, Stheta_only_translation2)
    pt.assert_exponential_coordinates_equal(
        Stheta_only_translation, Stheta_only_translation2
    )

    rng = np.random.default_rng(381)

    # 180 degree rotation ambiguity
    for _ in range(10):
        q = rng.standard_normal(size=3)
        s = pr.norm_vector(rng.standard_normal(size=3))
        h = rng.standard_normal()
        Stheta = pt.screw_axis_from_screw_parameters(q, s, h) * np.pi
        Stheta2 = pt.screw_axis_from_screw_parameters(q, -s, -h) * np.pi
        assert_array_almost_equal(
            pt.transform_from_exponential_coordinates(Stheta),
            pt.transform_from_exponential_coordinates(Stheta2),
        )
        assert_array_almost_equal(
            pt.norm_exponential_coordinates(Stheta),
            pt.norm_exponential_coordinates(Stheta2),
        )
        pt.assert_exponential_coordinates_equal(Stheta, Stheta2)

    for _ in range(10):
        Stheta = rng.standard_normal(size=6)
        # ensure that theta is not within [-pi, pi]
        i = rng.integers(0, 3)
        Stheta[i] = np.sign(Stheta[i]) * (np.pi + rng.random())
        Stheta_norm = pt.norm_exponential_coordinates(Stheta)
        assert not np.all(Stheta == Stheta_norm)
        pt.assert_exponential_coordinates_equal(Stheta, Stheta_norm)

        A2B = pt.transform_from_exponential_coordinates(Stheta)
        Stheta2 = pt.exponential_coordinates_from_transform(A2B)
        assert_array_almost_equal(Stheta2, Stheta_norm)


def test_assert_screw_parameters_equal():
    q = np.array([1, 2, 3])
    s_axis = pr.norm_vector(np.array([-2, 3, 5]))
    h = 2
    theta = 1
    pt.assert_screw_parameters_equal(
        q, s_axis, h, theta, q + 484.3 * s_axis, s_axis, h, theta
    )

    with pytest.raises(AssertionError):
        pt.assert_screw_parameters_equal(
            q, s_axis, h, theta, q + 484.3, s_axis, h, theta
        )

    s_axis_mirrored = -s_axis
    theta_mirrored = 2.0 * np.pi - theta
    h_mirrored = -h * theta / theta_mirrored
    pt.assert_screw_parameters_equal(
        q, s_axis_mirrored, h_mirrored, theta_mirrored, q, s_axis, h, theta
    )


def test_conversions_between_screw_axis_and_parameters():
    rng = np.random.default_rng(98)

    q = pr.random_vector(rng, 3)
    s_axis = pr.norm_vector(pr.random_vector(rng, 3))
    h = np.inf
    screw_axis = pt.screw_axis_from_screw_parameters(q, s_axis, h)
    assert_array_almost_equal(screw_axis, np.r_[0, 0, 0, s_axis])
    q2, s_axis2, h2 = pt.screw_parameters_from_screw_axis(screw_axis)

    assert_array_almost_equal(np.zeros(3), q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert_array_almost_equal(h, h2)

    for _ in range(10):
        s_axis = pr.norm_vector(pr.random_vector(rng, 3))
        # q has to be orthogonal to s_axis so that we reconstruct it exactly
        q = pr.perpendicular_to_vector(s_axis)
        h = rng.random() + 0.5

        screw_axis = pt.screw_axis_from_screw_parameters(q, s_axis, h)
        q2, s_axis2, h2 = pt.screw_parameters_from_screw_axis(screw_axis)

        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(s_axis, s_axis2)
        assert_array_almost_equal(h, h2)


def test_conversions_between_exponential_coordinates_and_transform():
    A2B = np.eye(4)
    Stheta = pt.exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    A2B2 = pt.transform_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)

    A2B = pt.translate_transform(np.eye(4), [1.0, 5.0, 0.0])
    Stheta = pt.exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 1.0, 5.0, 0.0])
    A2B2 = pt.transform_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)

    A2B = pt.rotate_transform(
        np.eye(4), pr.active_matrix_from_angle(2, 0.5 * np.pi)
    )
    Stheta = pt.exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.5 * np.pi, 0.0, 0.0, 0.0])
    A2B2 = pt.transform_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)

    rng = np.random.default_rng(52)
    for _ in range(5):
        A2B = pt.random_transform(rng)
        Stheta = pt.exponential_coordinates_from_transform(A2B)
        A2B2 = pt.transform_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(A2B, A2B2)


def test_transform_from_exponential_coordinates_without_check():
    Stheta = np.zeros(7)
    with pytest.raises(ValueError, match="could not broadcast input array"):
        pt.transform_from_exponential_coordinates(Stheta, check=False)


def test_conversions_between_screw_axis_and_exponential_coordinates():
    S = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    theta = 1.0
    Stheta = pt.exponential_coordinates_from_screw_axis(S, theta)
    S2, theta2 = pt.screw_axis_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(S, S2)
    assert pytest.approx(theta) == theta2

    S = np.zeros(6)
    theta = 0.0
    S2, theta2 = pt.screw_axis_from_exponential_coordinates(np.zeros(6))
    assert_array_almost_equal(S, S2)
    assert pytest.approx(theta) == theta2

    rng = np.random.default_rng(33)
    for _ in range(5):
        S = pt.random_screw_axis(rng)
        theta = rng.random()
        Stheta = pt.exponential_coordinates_from_screw_axis(S, theta)
        S2, theta2 = pt.screw_axis_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(S, S2)
        assert pytest.approx(theta) == theta2


def test_conversions_between_exponential_coordinates_and_transform_log():
    rng = np.random.default_rng(22)
    for _ in range(5):
        Stheta = rng.standard_normal(size=6)
        transform_log = pt.transform_log_from_exponential_coordinates(Stheta)
        Stheta2 = pt.exponential_coordinates_from_transform_log(transform_log)
        assert_array_almost_equal(Stheta, Stheta2)


def test_exponential_coordinates_from_transform_log_without_check():
    transform_log = np.ones((4, 4))
    Stheta = pt.exponential_coordinates_from_transform_log(
        transform_log, check=False
    )
    assert_array_almost_equal(Stheta, np.ones(6))


def test_conversions_between_screw_matrix_and_screw_axis():
    rng = np.random.default_rng(83)
    for _ in range(5):
        S = pt.random_screw_axis(rng)
        S_mat = pt.screw_matrix_from_screw_axis(S)
        S2 = pt.screw_axis_from_screw_matrix(S_mat)
        assert_array_almost_equal(S, S2)


def test_conversions_between_screw_matrix_and_transform_log():
    S_mat = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    theta = 1.0
    transform_log = pt.transform_log_from_screw_matrix(S_mat, theta)
    S_mat2, theta2 = pt.screw_matrix_from_transform_log(transform_log)
    assert_array_almost_equal(S_mat, S_mat2)
    assert pytest.approx(theta) == theta2

    S_mat = np.zeros((4, 4))
    theta = 0.0
    transform_log = pt.transform_log_from_screw_matrix(S_mat, theta)
    S_mat2, theta2 = pt.screw_matrix_from_transform_log(transform_log)
    assert_array_almost_equal(S_mat, S_mat2)
    assert pytest.approx(theta) == theta2

    rng = np.random.default_rng(65)
    for _ in range(5):
        S = pt.random_screw_axis(rng)
        theta = float(np.random.random())
        S_mat = pt.screw_matrix_from_screw_axis(S)
        transform_log = pt.transform_log_from_screw_matrix(S_mat, theta)
        S_mat2, theta2 = pt.screw_matrix_from_transform_log(transform_log)
        assert_array_almost_equal(S_mat, S_mat2)
        assert pytest.approx(theta) == theta2


def test_conversions_between_transform_and_transform_log():
    A2B = np.eye(4)
    transform_log = pt.transform_log_from_transform(A2B)
    assert_array_almost_equal(transform_log, np.zeros((4, 4)))
    A2B2 = pt.transform_from_transform_log(transform_log)
    assert_array_almost_equal(A2B, A2B2)

    rng = np.random.default_rng(84)
    A2B = pt.transform_from(np.eye(3), p=pr.random_vector(rng, 3))
    transform_log = pt.transform_log_from_transform(A2B)
    A2B2 = pt.transform_from_transform_log(transform_log)
    assert_array_almost_equal(A2B, A2B2)

    for _ in range(5):
        A2B = pt.random_transform(rng)
        transform_log = pt.transform_log_from_transform(A2B)
        A2B2 = pt.transform_from_transform_log(transform_log)
        assert_array_almost_equal(A2B, A2B2)


def test_dual_quaternion_from_screw_parameters():
    q = np.zeros(3)
    s_axis = np.array([1, 0, 0])
    h = np.inf
    theta = 0.0
    dq = pt.dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
    assert_array_almost_equal(dq, np.array([1, 0, 0, 0, 0, 0, 0, 0]))

    q = np.zeros(3)
    s_axis = pr.norm_vector(np.array([2.3, 2.4, 2.5]))
    h = np.inf
    theta = 3.6
    dq = pt.dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
    pq = pt.pq_from_dual_quaternion(dq)
    assert_array_almost_equal(pq, np.r_[s_axis * theta, 1, 0, 0, 0])

    q = np.zeros(3)
    s_axis = pr.norm_vector(np.array([2.4, 2.5, 2.6]))
    h = 0.0
    theta = 4.1
    dq = pt.dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
    pq = pt.pq_from_dual_quaternion(dq)
    assert_array_almost_equal(pq[:3], [0, 0, 0])
    assert_array_almost_equal(
        pr.axis_angle_from_quaternion(pq[3:]),
        pr.norm_axis_angle(np.r_[s_axis, theta]),
    )

    rng = np.random.default_rng(1001)
    for _ in range(5):
        A2B = pt.random_transform(rng)
        Stheta = pt.exponential_coordinates_from_transform(A2B)
        S, theta = pt.screw_axis_from_exponential_coordinates(Stheta)
        q, s_axis, h = pt.screw_parameters_from_screw_axis(S)
        dq = pt.dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
        pt.assert_unit_dual_quaternion(dq)

        dq_expected = pt.dual_quaternion_from_transform(A2B)
        pt.assert_unit_dual_quaternion_equal(dq, dq_expected)
