import warnings
import numpy as np
import pytest

from pytransform3d.transformations import (
    random_transform, transform_from, translate_transform, rotate_transform,
    invert_transform, vector_to_point, vectors_to_points, vector_to_direction,
    vectors_to_directions, concat, transform, scale_transform,
    assert_transform, check_transform, check_pq, pq_from_transform,
    transform_from_pq, check_screw_parameters, check_screw_axis,
    check_exponential_coordinates, screw_axis_from_screw_parameters,
    screw_parameters_from_screw_axis, transform_from_exponential_coordinates,
    exponential_coordinates_from_transform, random_screw_axis,
    exponential_coordinates_from_screw_axis,
    screw_axis_from_exponential_coordinates,
    transform_log_from_exponential_coordinates,
    exponential_coordinates_from_transform_log,
    screw_matrix_from_screw_axis, screw_axis_from_screw_matrix,
    transform_log_from_screw_matrix, screw_matrix_from_transform_log,
    transform_from_transform_log, transform_log_from_transform,
    check_screw_matrix, check_transform_log,
    adjoint_from_transform, norm_exponential_coordinates,
    dual_quaternion_from_transform, transform_from_dual_quaternion,
    concatenate_dual_quaternions, dq_prod_vector, check_dual_quaternion,
    assert_unit_dual_quaternion, dual_quaternion_from_pq,
    pq_from_dual_quaternion, assert_unit_dual_quaternion_equal,
    screw_parameters_from_dual_quaternion, assert_screw_parameters_equal,
    dual_quaternion_from_screw_parameters, dual_quaternion_sclerp)
from pytransform3d.rotations import (
    matrix_from_axis_angle, random_axis_angle, random_vector,
    axis_angle_from_matrix, norm_vector, perpendicular_to_vector,
    active_matrix_from_angle,
    random_quaternion, axis_angle_from_quaternion, norm_axis_angle)
from numpy.testing import assert_array_almost_equal


def test_check_transform():
    """Test input validation for transformation matrix."""
    A2B = np.eye(3)
    with pytest.raises(
            ValueError,
            match="Expected homogeneous transformation matrix with shape"):
        check_transform(A2B)

    A2B = np.eye(4, dtype=int)
    A2B = check_transform(A2B)
    assert type(A2B) == np.ndarray
    assert A2B.dtype == np.float64

    A2B[:3, :3] = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]])
    with pytest.raises(ValueError, match="rotation matrix"):
        check_transform(A2B)

    A2B = np.eye(4)
    A2B[3, :] = np.array([0.1, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="homogeneous transformation matrix"):
        check_transform(A2B)

    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)
    A2B2 = check_transform(A2B)
    assert_array_almost_equal(A2B, A2B2)


def test_translate_transform_with_check():
    A2B_broken = np.zeros((4, 4))
    with pytest.raises(ValueError, match="rotation matrix"):
        translate_transform(A2B_broken, np.zeros(3))


def test_rotate_transform_with_check():
    A2B_broken = np.zeros((4, 4))
    with pytest.raises(ValueError, match="rotation matrix"):
        rotate_transform(A2B_broken, np.eye(3))


def test_check_pq():
    """Test input validation for position and orientation quaternion."""
    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q2 = check_pq(q)
    assert_array_almost_equal(q, q2)

    q3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    q4 = check_pq(q3)
    assert_array_almost_equal(q3, q4)
    assert len(q3) == q4.shape[0]

    A2B = np.eye(4)
    with pytest.raises(ValueError,
                       match="position and orientation quaternion"):
        check_pq(A2B)
    q5 = np.zeros(8)
    with pytest.raises(
            ValueError, match="position and orientation quaternion"):
        check_pq(q5)


def test_invert_transform():
    """Test inversion of transformations."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        R = matrix_from_axis_angle(random_axis_angle(random_state))
        p = random_vector(random_state)
        A2B = transform_from(R, p)
        B2A = invert_transform(A2B)
        A2B2 = np.linalg.inv(B2A)
        assert_array_almost_equal(A2B, A2B2)


def test_invert_transform_without_check():
    """Test inversion of transformations."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        A2B = random_state.randn(4, 4)
        A2B = A2B + A2B.T
        B2A = invert_transform(A2B, check=False)
        A2B2 = np.linalg.inv(B2A)
        assert_array_almost_equal(A2B, A2B2)


def test_vector_to_point():
    """Test conversion from vector to homogenous coordinates."""
    v = np.array([1, 2, 3])
    pA = vector_to_point(v)
    assert_array_almost_equal(pA, [1, 2, 3, 1])

    random_state = np.random.RandomState(0)
    R = matrix_from_axis_angle(random_axis_angle(random_state))
    p = random_vector(random_state)
    A2B = transform_from(R, p)
    assert_transform(A2B)
    _ = transform(A2B, pA)


def test_vectors_to_points():
    """Test conversion from vectors to homogenous coordinates."""
    V = np.array([[1, 2, 3], [2, 3, 4]])
    PA = vectors_to_points(V)
    assert_array_almost_equal(PA, [[1, 2, 3, 1], [2, 3, 4, 1]])

    random_state = np.random.RandomState(0)
    V = random_state.randn(10, 3)
    for i, p in enumerate(vectors_to_points(V)):
        assert_array_almost_equal(p, vector_to_point(V[i]))


def test_vector_to_direction():
    """Test conversion from vector to direction in homogenous coordinates."""
    v = np.array([1, 2, 3])
    dA = vector_to_direction(v)
    assert_array_almost_equal(dA, [1, 2, 3, 0])

    random_state = np.random.RandomState(0)
    R = matrix_from_axis_angle(random_axis_angle(random_state))
    p = random_vector(random_state)
    A2B = transform_from(R, p)
    assert_transform(A2B)
    _ = transform(A2B, dA)


def test_vectors_to_directions():
    """Test conversion from vectors to directions in homogenous coordinates."""
    V = np.array([[1, 2, 3], [2, 3, 4]])
    DA = vectors_to_directions(V)
    assert_array_almost_equal(DA, [[1, 2, 3, 0], [2, 3, 4, 0]])

    random_state = np.random.RandomState(0)
    V = random_state.randn(10, 3)
    for i, d in enumerate(vectors_to_directions(V)):
        assert_array_almost_equal(d, vector_to_direction(V[i]))


def test_concat():
    """Test concatenation of transforms."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        A2B = random_transform(random_state)
        assert_transform(A2B)

        B2C = random_transform(random_state)
        assert_transform(B2C)

        A2C = concat(A2B, B2C)
        assert_transform(A2C)

        p_A = np.array([0.3, -0.2, 0.9, 1.0])
        p_C = transform(A2C, p_A)

        C2A = invert_transform(A2C)
        p_A2 = transform(C2A, p_C)

        assert_array_almost_equal(p_A, p_A2)

        C2A2 = concat(invert_transform(B2C), invert_transform(A2B))
        p_A3 = transform(C2A2, p_C)
        assert_array_almost_equal(p_A, p_A3)


def test_transform():
    """Test transformation of points."""
    PA = np.array([[1, 2, 3, 1],
                   [2, 3, 4, 1]])

    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)

    PB = transform(A2B, PA)
    p0B = transform(A2B, PA[0])
    p1B = transform(A2B, PA[1])
    assert_array_almost_equal(PB, np.array([p0B, p1B]))

    with pytest.raises(
            ValueError,
            match="Cannot transform array with more than 2 dimensions"):
        transform(A2B, np.zeros((2, 2, 4)))


def test_scale_transform():
    """Test scaling of transforms."""
    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)

    A2B_scaled1 = scale_transform(A2B, s_xt=0.5, s_yt=0.5, s_zt=0.5)
    A2B_scaled2 = scale_transform(A2B, s_t=0.5)
    assert_array_almost_equal(A2B_scaled1, A2B_scaled2)

    A2B_scaled1 = scale_transform(A2B, s_xt=0.5, s_yt=0.5, s_zt=0.5, s_r=0.5)
    A2B_scaled2 = scale_transform(A2B, s_t=0.5, s_r=0.5)
    A2B_scaled3 = scale_transform(A2B, s_d=0.5)
    assert_array_almost_equal(A2B_scaled1, A2B_scaled2)
    assert_array_almost_equal(A2B_scaled1, A2B_scaled3)

    A2B_scaled = scale_transform(A2B, s_xr=0.0)
    a_scaled = axis_angle_from_matrix(A2B_scaled[:3, :3])
    assert_array_almost_equal(a_scaled[0], 0.0)
    A2B_scaled = scale_transform(A2B, s_yr=0.0)
    a_scaled = axis_angle_from_matrix(A2B_scaled[:3, :3])
    assert_array_almost_equal(a_scaled[1], 0.0)
    A2B_scaled = scale_transform(A2B, s_zr=0.0)
    a_scaled = axis_angle_from_matrix(A2B_scaled[:3, :3])
    assert_array_almost_equal(a_scaled[2], 0.0)


def test_pq_from_transform():
    """Test conversion from homogeneous matrix to position and quaternion."""
    A2B = np.eye(4)
    pq = pq_from_transform(A2B)
    assert_array_almost_equal(
        pq, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))


def test_transform_from_pq():
    """Test conversion from position and quaternion to homogeneous matrix."""
    pq = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    A2B = transform_from_pq(pq)
    assert_array_almost_equal(A2B, np.eye(4))


def test_deactivate_transform_precision_error():
    A2B = np.eye(4)
    A2B[0, 0] = 2.0
    A2B[3, 0] = 3.0
    with pytest.raises(ValueError, match="Expected rotation matrix"):
        check_transform(A2B)

    n_expected_warnings = 2
    try:
        warnings.filterwarnings("always", category=UserWarning)
        with warnings.catch_warnings(record=True) as w:
            check_transform(A2B, strict_check=False)
            assert len(w) == n_expected_warnings
    finally:
        warnings.filterwarnings("default", category=UserWarning)


def test_norm_exponential_coordinates():
    Stheta_only_translation = np.array([0.0, 0.0, 0.0, 100.0, 25.0, -23.0])
    Stheta_only_translation2 = norm_exponential_coordinates(
        Stheta_only_translation)
    assert_array_almost_equal(
        Stheta_only_translation, Stheta_only_translation2)

    random_state = np.random.RandomState(381)

    # 180 degree rotation ambiguity
    for _ in range(10):
        q = random_state.randn(3)
        s = norm_vector(random_state.randn(3))
        h = random_state.randn()
        Stheta = screw_axis_from_screw_parameters(q, s, h) * np.pi
        Stheta2 = screw_axis_from_screw_parameters(q, -s, -h) * np.pi
        assert_array_almost_equal(
            transform_from_exponential_coordinates(Stheta),
            transform_from_exponential_coordinates(Stheta2))
        assert_array_almost_equal(norm_exponential_coordinates(Stheta),
                                  norm_exponential_coordinates(Stheta2))

    for _ in range(10):
        Stheta = random_state.randn(6)
        # ensure that theta is not within [-pi, pi]
        Stheta[random_state.randint(0, 3)] += np.pi + random_state.rand()
        Stheta_norm = norm_exponential_coordinates(Stheta)
        assert not np.all(Stheta == Stheta_norm)

        A2B = transform_from_exponential_coordinates(Stheta)
        Stheta2 = exponential_coordinates_from_transform(A2B)
        assert_array_almost_equal(Stheta2, Stheta_norm)


def test_check_screw_parameters():
    q = [100.0, -20.3, 1e-3]
    s_axis = norm_vector(np.array([-1.0, 2.0, 3.0])).tolist()
    h = 0.2

    with pytest.raises(ValueError, match="Expected 3D vector with shape"):
        check_screw_parameters([0.0], s_axis, h)
    with pytest.raises(ValueError, match="Expected 3D vector with shape"):
        check_screw_parameters(q, [0.0], h)
    with pytest.raises(ValueError, match="s_axis must not have norm 0"):
        check_screw_parameters(q, np.zeros(3), h)

    q2, s_axis2, h2 = check_screw_parameters(q, 2.0 * np.array(s_axis), h)
    assert_array_almost_equal(q, q2)
    assert pytest.approx(h) == h2
    assert pytest.approx(np.linalg.norm(s_axis2)) == 1.0

    q2, s_axis2, h2 = check_screw_parameters(q, s_axis, h)
    assert_array_almost_equal(q, q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert pytest.approx(h) == h2

    q2, s_axis2, h2 = check_screw_parameters(q, s_axis, np.inf)
    assert_array_almost_equal(np.zeros(3), q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert np.isinf(h2)


def test_check_screw_axis():
    random_state = np.random.RandomState(73)
    omega = norm_vector(random_vector(random_state, 3))
    v = random_vector(random_state, 3)

    with pytest.raises(ValueError, match="Expected 3D vector with shape"):
        check_screw_axis(np.r_[0, 1, v])

    with pytest.raises(
            ValueError, match="Norm of rotation axis must either be 0 or 1"):
        check_screw_axis(np.r_[2 * omega, v])

    with pytest.raises(
            ValueError, match="If the norm of the rotation axis is 0"):
        check_screw_axis(np.r_[0, 0, 0, v])

    S_pure_translation = np.r_[0, 0, 0, norm_vector(v)]
    S = check_screw_axis(S_pure_translation)
    assert_array_almost_equal(S, S_pure_translation)

    S_both = np.r_[omega, v]
    S = check_screw_axis(S_both)
    assert_array_almost_equal(S, S_both)


def test_check_exponential_coordinates():
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_exponential_coordinates([0])

    Stheta = [0.0, 1.0, 2.0, -5.0, -2, 3]
    Stheta2 = check_exponential_coordinates(Stheta)
    assert_array_almost_equal(Stheta, Stheta2)


def test_check_screw_matrix():
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_screw_matrix(np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_screw_matrix(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_screw_matrix(np.zeros((4, 3)))

    with pytest.raises(ValueError, match="Last row of screw matrix must only "
                                         "contains zeros"):
        check_screw_matrix(np.eye(4))

    screw_matrix = screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])) * 1.1
    with pytest.raises(
            ValueError, match="Norm of rotation axis must either be 0 or 1"):
        check_screw_matrix(screw_matrix)

    screw_matrix = screw_matrix_from_screw_axis(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])) * 1.1
    with pytest.raises(
            ValueError, match="If the norm of the rotation axis is 0"):
        check_screw_matrix(screw_matrix)

    screw_matrix = screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    screw_matrix2 = check_screw_matrix(screw_matrix)
    assert_array_almost_equal(screw_matrix, screw_matrix2)

    screw_matrix = screw_matrix_from_screw_axis(
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    screw_matrix2 = check_screw_matrix(screw_matrix)
    assert_array_almost_equal(screw_matrix, screw_matrix2)

    screw_matrix = screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    screw_matrix[0, 0] = 0.0001
    with pytest.raises(ValueError, match="Expected skew-symmetric matrix"):
        check_screw_matrix(screw_matrix)


def test_check_transform_log():
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_transform_log(np.zeros((1, 4, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_transform_log(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="Expected array-like with shape"):
        check_transform_log(np.zeros((4, 3)))

    with pytest.raises(
            ValueError, match="Last row of logarithm of transformation must "
                              "only contains zeros"):
        check_transform_log(np.eye(4))
    transform_log = screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])) * 1.1
    transform_log2 = check_transform_log(transform_log)
    assert_array_almost_equal(transform_log, transform_log2)

    transform_log = screw_matrix_from_screw_axis(
        np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])) * 1.1
    transform_log[0, 0] = 0.0001
    with pytest.raises(ValueError, match="Expected skew-symmetric matrix"):
        check_transform_log(transform_log)


def test_random_screw_axis():
    random_state = np.random.RandomState(893)
    for _ in range(5):
        S = random_screw_axis(random_state)
        check_screw_axis(S)


def test_conversions_between_screw_axis_and_parameters():
    random_state = np.random.RandomState(98)

    q = random_vector(random_state, 3)
    s_axis = norm_vector(random_vector(random_state, 3))
    h = np.inf
    screw_axis = screw_axis_from_screw_parameters(q, s_axis, h)
    assert_array_almost_equal(screw_axis, np.r_[0, 0, 0, s_axis])
    q2, s_axis2, h2 = screw_parameters_from_screw_axis(screw_axis)

    assert_array_almost_equal(np.zeros(3), q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert_array_almost_equal(h, h2)

    for _ in range(10):
        s_axis = norm_vector(random_vector(random_state, 3))
        # q has to be orthogonal to s_axis so that we reconstruct it exactly
        q = perpendicular_to_vector(s_axis)
        h = random_state.rand() + 0.5

        screw_axis = screw_axis_from_screw_parameters(q, s_axis, h)
        q2, s_axis2, h2 = screw_parameters_from_screw_axis(screw_axis)

        assert_array_almost_equal(q, q2)
        assert_array_almost_equal(s_axis, s_axis2)
        assert_array_almost_equal(h, h2)


def test_conversions_between_exponential_coordinates_and_transform():
    A2B = np.eye(4)
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    A2B2 = transform_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)

    A2B = translate_transform(np.eye(4), [1.0, 5.0, 0.0])
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.0, 1.0, 5.0, 0.0])
    A2B2 = transform_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)

    A2B = rotate_transform(np.eye(4), active_matrix_from_angle(2, 0.5 * np.pi))
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(Stheta, [0.0, 0.0, 0.5 * np.pi, 0.0, 0.0, 0.0])
    A2B2 = transform_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(A2B, A2B2)

    random_state = np.random.RandomState(52)
    for _ in range(5):
        A2B = random_transform(random_state)
        Stheta = exponential_coordinates_from_transform(A2B)
        A2B2 = transform_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(A2B, A2B2)


def test_conversions_between_screw_axis_and_exponential_coordinates():
    S = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    theta = 1.0
    Stheta = exponential_coordinates_from_screw_axis(S, theta)
    S2, theta2 = screw_axis_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(S, S2)
    assert pytest.approx(theta) == theta2

    S = np.zeros(6)
    theta = 0.0
    S2, theta2 = screw_axis_from_exponential_coordinates(np.zeros(6))
    assert_array_almost_equal(S, S2)
    assert pytest.approx(theta) == theta2

    random_state = np.random.RandomState(33)
    for _ in range(5):
        S = random_screw_axis(random_state)
        theta = random_state.rand()
        Stheta = exponential_coordinates_from_screw_axis(S, theta)
        S2, theta2 = screw_axis_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(S, S2)
        assert pytest.approx(theta) == theta2


def test_conversions_between_exponential_coordinates_and_transform_log():
    random_state = np.random.RandomState(22)
    for _ in range(5):
        Stheta = random_state.randn(6)
        transform_log = transform_log_from_exponential_coordinates(Stheta)
        Stheta2 = exponential_coordinates_from_transform_log(transform_log)
        assert_array_almost_equal(Stheta, Stheta2)


def test_conversions_between_screw_matrix_and_screw_axis():
    random_state = np.random.RandomState(83)
    for _ in range(5):
        S = random_screw_axis(random_state)
        S_mat = screw_matrix_from_screw_axis(S)
        S2 = screw_axis_from_screw_matrix(S_mat)
        assert_array_almost_equal(S, S2)


def test_conversions_between_screw_matrix_and_transform_log():
    S_mat = np.array([[0.0, 0.0, 0.0, 1.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
    theta = 1.0
    transform_log = transform_log_from_screw_matrix(S_mat, theta)
    S_mat2, theta2 = screw_matrix_from_transform_log(transform_log)
    assert_array_almost_equal(S_mat, S_mat2)
    assert pytest.approx(theta) == theta2

    S_mat = np.zeros((4, 4))
    theta = 0.0
    transform_log = transform_log_from_screw_matrix(S_mat, theta)
    S_mat2, theta2 = screw_matrix_from_transform_log(transform_log)
    assert_array_almost_equal(S_mat, S_mat2)
    assert pytest.approx(theta) == theta2

    random_state = np.random.RandomState(65)
    for _ in range(5):
        S = random_screw_axis(random_state)
        theta = float(np.random.rand())
        S_mat = screw_matrix_from_screw_axis(S)
        transform_log = transform_log_from_screw_matrix(S_mat, theta)
        S_mat2, theta2 = screw_matrix_from_transform_log(transform_log)
        assert_array_almost_equal(S_mat, S_mat2)
        assert pytest.approx(theta) == theta2


def test_conversions_between_transform_and_transform_log():
    A2B = np.eye(4)
    transform_log = transform_log_from_transform(A2B)
    assert_array_almost_equal(transform_log, np.zeros((4, 4)))
    A2B2 = transform_from_transform_log(transform_log)
    assert_array_almost_equal(A2B, A2B2)

    random_state = np.random.RandomState(84)
    A2B = transform_from(np.eye(3), p=random_vector(random_state, 3))
    transform_log = transform_log_from_transform(A2B)
    A2B2 = transform_from_transform_log(transform_log)
    assert_array_almost_equal(A2B, A2B2)

    for _ in range(5):
        A2B = random_transform(random_state)
        transform_log = transform_log_from_transform(A2B)
        A2B2 = transform_from_transform_log(transform_log)
        assert_array_almost_equal(A2B, A2B2)


def test_adjoint_of_transformation():
    random_state = np.random.RandomState(94)
    for _ in range(5):
        A2B = random_transform(random_state)
        theta_dot = 3.0 * float(random_state.rand())
        S = random_screw_axis(random_state)

        V_A = S * theta_dot

        adj_A2B = adjoint_from_transform(A2B)
        V_B = adj_A2B.dot(V_A)

        S_mat = screw_matrix_from_screw_axis(S)
        V_mat_A = S_mat * theta_dot
        V_mat_B = np.dot(np.dot(A2B, V_mat_A), invert_transform(A2B))

        S_B, theta_dot2 = screw_axis_from_exponential_coordinates(V_B)
        V_mat_B2 = screw_matrix_from_screw_axis(S_B) * theta_dot2
        assert pytest.approx(theta_dot) == theta_dot2
        assert_array_almost_equal(V_mat_B, V_mat_B2)


def test_check_dual_quaternion():
    dq1 = [0, 0]
    with pytest.raises(ValueError, match="Expected dual quaternion"):
        check_dual_quaternion(dq1)

    dq2 = [[0, 0]]
    with pytest.raises(ValueError, match="Expected dual quaternion"):
        check_dual_quaternion(dq2)

    dq3 = check_dual_quaternion([0] * 8, unit=False)
    assert dq3.shape[0] == 8


def test_normalize_dual_quaternion():
    dq = [1, 0, 0, 0, 0, 0, 0, 0]
    dq_norm = check_dual_quaternion(dq)
    assert_unit_dual_quaternion(dq_norm)
    assert_array_almost_equal(dq, dq_norm)

    dq = [0, 0, 0, 0, 0, 0, 0, 0]
    dq_norm = check_dual_quaternion(dq)
    assert_unit_dual_quaternion(dq_norm)
    assert_array_almost_equal([1, 0, 0, 0, 0, 0, 0, 0], dq_norm)

    random_state = np.random.RandomState(999)
    for _ in range(5):
        A2B = random_transform(random_state)
        dq = random_state.randn() * dual_quaternion_from_transform(A2B)
        dq_norm = check_dual_quaternion(dq)
        assert_unit_dual_quaternion(dq_norm)


def test_conversions_between_dual_quternion_and_transform():
    random_state = np.random.RandomState(1000)
    for _ in range(5):
        A2B = random_transform(random_state)
        dq = dual_quaternion_from_transform(A2B)
        A2B2 = transform_from_dual_quaternion(dq)
        assert_array_almost_equal(A2B, A2B2)
        dq2 = dual_quaternion_from_transform(A2B2)
        assert_unit_dual_quaternion_equal(dq, dq2)
    for _ in range(5):
        p = random_vector(random_state, 3)
        q = random_quaternion(random_state)
        dq = dual_quaternion_from_pq(np.hstack((p, q)))
        A2B = transform_from_dual_quaternion(dq)
        dq2 = dual_quaternion_from_transform(A2B)
        assert_unit_dual_quaternion_equal(dq, dq2)
        A2B2 = transform_from_dual_quaternion(dq2)
        assert_array_almost_equal(A2B, A2B2)


def test_conversions_between_dual_quternion_and_pq():
    random_state = np.random.RandomState(1000)
    for _ in range(5):
        pq = random_vector(random_state, 7)
        pq[3:] /= np.linalg.norm(pq[3:])
        dq = dual_quaternion_from_pq(pq)
        pq2 = pq_from_dual_quaternion(dq)
        assert_array_almost_equal(pq, pq2)
        dq2 = dual_quaternion_from_pq(pq2)
        assert_unit_dual_quaternion_equal(dq, dq2)


def test_dual_quaternion_concatenation():
    random_state = np.random.RandomState(1000)
    for _ in range(5):
        A2B = random_transform(random_state)
        B2C = random_transform(random_state)
        A2C = concat(A2B, B2C)
        dq1 = dual_quaternion_from_transform(A2B)
        dq2 = dual_quaternion_from_transform(B2C)
        dq3 = concatenate_dual_quaternions(dq2, dq1)
        A2C2 = transform_from_dual_quaternion(dq3)
        assert_array_almost_equal(A2C, A2C2)


def test_dual_quaternion_applied_to_point():
    random_state = np.random.RandomState(1000)
    for _ in range(5):
        p_A = random_vector(random_state, 3)
        A2B = random_transform(random_state)
        dq = dual_quaternion_from_transform(A2B)
        p_B = dq_prod_vector(dq, p_A)
        assert_array_almost_equal(
            p_B, transform(A2B, vector_to_point(p_A))[:3])


def test_assert_screw_parameters_equal():
    q = np.array([1, 2, 3])
    s_axis = norm_vector(np.array([-2, 3, 5]))
    h = 2
    theta = 1
    assert_screw_parameters_equal(
        q, s_axis, h, theta,
        q + 484.3 * s_axis, s_axis, h, theta)

    with pytest.raises(AssertionError):
        assert_screw_parameters_equal(
            q, s_axis, h, theta, q + 484.3, s_axis, h, theta)

    s_axis_mirrored = -s_axis
    theta_mirrored = 2.0 * np.pi - theta
    h_mirrored = -h * theta / theta_mirrored
    assert_screw_parameters_equal(
        q, s_axis_mirrored, h_mirrored, theta_mirrored,
        q, s_axis, h, theta)


def test_screw_parameters_from_dual_quaternion():
    dq = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    q, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    assert_array_almost_equal(q, np.zeros(3))
    assert_array_almost_equal(s_axis, np.array([1, 0, 0]))
    assert np.isinf(h)
    assert pytest.approx(theta) == 0

    dq = dual_quaternion_from_pq(np.array([1.2, 1.3, 1.4, 1, 0, 0, 0]))
    q, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    assert_array_almost_equal(q, np.zeros(3))
    assert_array_almost_equal(s_axis, norm_vector(np.array([1.2, 1.3, 1.4])))
    assert np.isinf(h)
    assert pytest.approx(theta) == np.linalg.norm(np.array([1.2, 1.3, 1.4]))

    random_state = np.random.RandomState(1001)
    quat = random_quaternion(random_state)
    a = axis_angle_from_quaternion(quat)
    dq = dual_quaternion_from_pq(
        np.r_[0, 0, 0, quat])
    q, s_axis, h, theta = screw_parameters_from_dual_quaternion(dq)
    assert_array_almost_equal(q, np.zeros(3))
    assert_array_almost_equal(s_axis, a[:3])
    assert_array_almost_equal(h, 0)
    assert_array_almost_equal(theta, a[3])

    for _ in range(5):
        A2B = random_transform(random_state)
        dq = dual_quaternion_from_transform(A2B)
        Stheta = exponential_coordinates_from_transform(A2B)
        S, theta = screw_axis_from_exponential_coordinates(Stheta)
        q, s_axis, h = screw_parameters_from_screw_axis(S)
        q2, s_axis2, h2, theta2 = screw_parameters_from_dual_quaternion(dq)
        assert_screw_parameters_equal(
            q, s_axis, h, theta, q2, s_axis2, h2, theta2)


def test_dual_quaternion_from_screw_parameters():
    q = np.zeros(3)
    s_axis = np.array([1, 0, 0])
    h = np.inf
    theta = 0.0
    dq = dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
    assert_array_almost_equal(dq, np.array([1, 0, 0, 0, 0, 0, 0, 0]))

    q = np.zeros(3)
    s_axis = norm_vector(np.array([2.3, 2.4, 2.5]))
    h = np.inf
    theta = 3.6
    dq = dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
    pq = pq_from_dual_quaternion(dq)
    assert_array_almost_equal(pq, np.r_[s_axis * theta, 1, 0, 0, 0])

    q = np.zeros(3)
    s_axis = norm_vector(np.array([2.4, 2.5, 2.6]))
    h = 0.0
    theta = 4.1
    dq = dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
    pq = pq_from_dual_quaternion(dq)
    assert_array_almost_equal(pq[:3], [0, 0, 0])
    assert_array_almost_equal(
        axis_angle_from_quaternion(pq[3:]),
        norm_axis_angle(np.r_[s_axis, theta]))

    random_state = np.random.RandomState(1001)
    for _ in range(5):
        A2B = random_transform(random_state)
        Stheta = exponential_coordinates_from_transform(A2B)
        S, theta = screw_axis_from_exponential_coordinates(Stheta)
        q, s_axis, h = screw_parameters_from_screw_axis(S)
        dq = dual_quaternion_from_screw_parameters(q, s_axis, h, theta)
        assert_unit_dual_quaternion(dq)

        dq_expected = dual_quaternion_from_transform(A2B)
        assert_unit_dual_quaternion_equal(dq, dq_expected)


def test_dual_quaternion_sclerp_same_dual_quaternions():
    random_state = np.random.RandomState(19)
    pose = random_transform(random_state)
    dq = dual_quaternion_from_transform(pose)
    dq2 = dual_quaternion_sclerp(dq, dq, 0.5)
    assert_array_almost_equal(dq, dq2)


def test_dual_quaternion_sclerp():
    random_state = np.random.RandomState(22)
    pose1 = random_transform(random_state)
    pose2 = random_transform(random_state)
    dq1 = dual_quaternion_from_transform(pose1)
    dq2 = dual_quaternion_from_transform(pose2)

    n_steps = 100

    # Ground truth: screw linear interpolation
    pose12pose2 = concat(pose2, invert_transform(pose1))
    screw_axis, theta = screw_axis_from_exponential_coordinates(
        exponential_coordinates_from_transform(pose12pose2))
    offsets = np.array(
        [transform_from_exponential_coordinates(screw_axis * t * theta)
         for t in np.linspace(0, 1, n_steps)])
    interpolated_poses = np.array([
        concat(offset, pose1) for offset in offsets])

    # Dual quaternion ScLERP
    sclerp_interpolated_dqs = np.vstack([
        dual_quaternion_sclerp(dq1, dq2, t)
        for t in np.linspace(0, 1, n_steps)])
    sclerp_interpolated_poses_from_dqs = np.array([
        transform_from_dual_quaternion(dq) for dq in sclerp_interpolated_dqs])

    for t in range(n_steps):
        assert_array_almost_equal(
            interpolated_poses[t], sclerp_interpolated_poses_from_dqs[t])


def test_dual_quaternion_sclerp_sign_ambiguity():
    n_steps = 10
    random_state = np.random.RandomState(2323)
    T1 = random_transform(random_state)
    dq1 = dual_quaternion_from_transform(T1)
    dq2 = np.copy(dq1)

    if np.sign(dq1[0]) != np.sign(dq2[0]):
        dq2 *= -1.0
    traj_q = [dual_quaternion_sclerp(dq1, dq2, t)
              for t in np.linspace(0, 1, n_steps)]
    path_length = np.sum([np.linalg.norm(r - s)
                          for r, s in zip(traj_q[:-1], traj_q[1:])])

    dq2 *= -1.0
    traj_q_opposing = [dual_quaternion_sclerp(dq1, dq2, t)
                       for t in np.linspace(0, 1, n_steps)]
    path_length_opposing = np.sum(
        [np.linalg.norm(r - s)
         for r, s in zip(traj_q_opposing[:-1], traj_q_opposing[1:])])

    assert path_length_opposing == path_length


def test_exponential_coordinates_from_almost_identity_transform():
    A2B = np.array([
        [0.9999999999999999, -1.5883146449068575e-16, 4.8699079321578667e-17,
         -7.54265065748827e-05],
        [5.110044286978025e-17, 0.9999999999999999, 1.1798895336935056e-17,
         9.340523179823812e-05],
        [3.0048299647976294e-18, 5.4741890703482423e-17, 1.0,
         -7.803584869947588e-05],
        [0, 0, 0, 1]])
    Stheta = exponential_coordinates_from_transform(A2B)
    assert_array_almost_equal(np.zeros(6), Stheta, decimal=4)


def test_transform_log_from_almost_identity_transform():
    A2B = np.array([
        [0.9999999999999999, -1.5883146449068575e-16, 4.8699079321578667e-17,
         -7.54265065748827e-05],
        [5.110044286978025e-17, 0.9999999999999999, 1.1798895336935056e-17,
         9.340523179823812e-05],
        [3.0048299647976294e-18, 5.4741890703482423e-17, 1.0,
         -7.803584869947588e-05],
        [0, 0, 0, 1]])
    transform_log = transform_log_from_transform(A2B)
    assert_array_almost_equal(np.zeros((4, 4)), transform_log)


def test_exponential_coordinates_from_transform_log_without_check():
    transform_log = np.ones((4, 4))
    Stheta = exponential_coordinates_from_transform_log(
        transform_log, check=False)
    assert_array_almost_equal(Stheta, np.ones(6))


def test_exponential_coordinates_from_transform_without_check():
    transform = np.ones((4, 4))
    Stheta = exponential_coordinates_from_transform(transform, check=False)
    assert_array_almost_equal(Stheta, np.array([0, 0, 0, 1, 1, 1]))


def test_transform_from_exponential_coordinates_without_check():
    Stheta = np.zeros(7)
    with pytest.raises(ValueError, match="could not broadcast input array"):
        transform_from_exponential_coordinates(Stheta, check=False)


def test_adjoint_from_transform_without_check():
    transform = np.ones((4, 4))
    adjoint = adjoint_from_transform(transform, check=False)
    assert_array_almost_equal(adjoint[:3, :3], np.ones((3, 3)))
    assert_array_almost_equal(adjoint[3:, 3:], np.ones((3, 3)))
    assert_array_almost_equal(adjoint[3:, :3], np.zeros((3, 3)))
    assert_array_almost_equal(adjoint[:3, 3:], np.zeros((3, 3)))
