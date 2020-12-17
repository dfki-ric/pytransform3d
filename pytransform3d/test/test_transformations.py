import warnings
import platform
import numpy as np
from pytransform3d.transformations import (random_transform, transform_from,
                                           translate_transform, rotate_transform,
                                           invert_transform, vector_to_point,
                                           vectors_to_points, vector_to_direction,
                                           vectors_to_directions,
                                           concat, transform, scale_transform,
                                           assert_transform, check_transform,
                                           check_pq, pq_from_transform,
                                           transform_from_pq,
                                           check_screw_parameters,
                                           check_screw_axis,
                                           check_exponential_coordinates,
                                           screw_axis_from_screw_parameters,
                                           screw_parameters_from_screw_axis,
                                           transform_from_exponential_coordinates,
                                           exponential_coordinates_from_transform,
                                           random_screw_axis,
                                           exponential_coordinates_from_screw_axis,
                                           screw_axis_from_exponential_coordinates,
                                           transform_log_from_exponential_coordinates,
                                           exponential_coordinates_from_transform_log,
                                           unit_twist_from_screw_axis,
                                           screw_axis_from_unit_twist,
                                           transform_log_from_unit_twist,
                                           unit_twist_from_transform_log)
from pytransform3d.rotations import (matrix_from, random_axis_angle,
                                     random_vector, axis_angle_from_matrix,
                                     norm_vector, perpendicular_to_vector,
                                     active_matrix_from_angle)
from nose.tools import assert_equal, assert_almost_equal, assert_raises_regexp
from numpy.testing import assert_array_almost_equal


def test_check_transform():
    """Test input validation for transformation matrix."""
    A2B = np.eye(3)
    assert_raises_regexp(
        ValueError, "Expected homogeneous transformation matrix with shape",
        check_transform, A2B)

    A2B = np.eye(4, dtype=int)
    A2B = check_transform(A2B)
    assert_equal(type(A2B), np.ndarray)
    assert_equal(A2B.dtype, np.float)

    A2B[:3, :3] = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]])
    assert_raises_regexp(ValueError, "rotation matrix", check_transform, A2B)

    A2B = np.eye(4)
    A2B[3, :] = np.array([0.1, 0.0, 0.0, 1.0])
    assert_raises_regexp(ValueError, "homogeneous transformation matrix",
                         check_transform, A2B)

    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)
    A2B2 = check_transform(A2B)
    assert_array_almost_equal(A2B, A2B2)


def test_translate_transform_with_check():
    A2B_broken = np.zeros((4, 4))
    assert_raises_regexp(
        ValueError, "rotation matrix", translate_transform,
        A2B_broken, np.zeros(3))


def test_rotate_transform_with_check():
    A2B_broken = np.zeros((4, 4))
    assert_raises_regexp(
        ValueError, "rotation matrix", rotate_transform,
        A2B_broken, np.eye(3))


def test_check_pq():
    """Test input validation for position and orientation quaternion."""
    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q2 = check_pq(q)
    assert_array_almost_equal(q, q2)

    q = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    q2 = check_pq(q)
    assert_array_almost_equal(q, q2)
    assert_equal(len(q), q2.shape[0])

    A2B = np.eye(4)
    assert_raises_regexp(ValueError, "position and orientation quaternion",
                         check_pq, A2B)
    q = np.zeros(8)
    assert_raises_regexp(ValueError, "position and orientation quaternion",
                         check_pq, q)


def test_invert_transform():
    """Test inversion of transformations."""
    random_state = np.random.RandomState(0)
    for _ in range(5):
        R = matrix_from(a=random_axis_angle(random_state))
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
    R = matrix_from(a=random_axis_angle(random_state))
    p = random_vector(random_state)
    A2B = transform_from(R, p)
    assert_transform(A2B)
    pB = transform(A2B, pA)


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
    R = matrix_from(a=random_axis_angle(random_state))
    p = random_vector(random_state)
    A2B = transform_from(R, p)
    assert_transform(A2B)
    dB = transform(A2B, dA)


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

    assert_raises_regexp(
        ValueError, "Cannot transform array with more than 2 dimensions",
        transform, A2B, np.zeros((2, 2, 4)))


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
    assert_array_almost_equal(pq, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))


def test_transform_from_pq():
    """Test conversion from position and quaternion to homogeneous matrix."""
    pq = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    A2B = transform_from_pq(pq)
    assert_array_almost_equal(A2B, np.eye(4))


def test_deactivate_transform_precision_error():
    A2B = np.eye(4)
    A2B[0, 0] = 2.0
    A2B[3, 0] = 3.0
    assert_raises_regexp(
        ValueError, "Expected rotation matrix", check_transform, A2B)

    if int(platform.python_version()[0]) == 2:
        # Python 2 seems to incorrectly suppress some warnings, not sure why
        n_expected_warnings = 2
    else:
        n_expected_warnings = 3
    try:
        warnings.filterwarnings("always", category=UserWarning)
        with warnings.catch_warnings(record=True) as w:
            check_transform(A2B, strict_check=False)
            assert_equal(len(w), n_expected_warnings)
    finally:
        warnings.filterwarnings("default", category=UserWarning)


def test_check_screw_parameters():
    q = [100.0, -20.3, 1e-3]
    s_axis = norm_vector(np.array([-1.0, 2.0, 3.0])).tolist()
    h = 0.2

    assert_raises_regexp(
        ValueError, "Expected 3D vector with shape",
        check_screw_parameters, [0.0], s_axis, h)
    assert_raises_regexp(
        ValueError, "Expected 3D vector with shape",
        check_screw_parameters, q, [0.0], h)
    assert_raises_regexp(
        ValueError, "s_axis must not have norm 0",
        check_screw_parameters, q, np.zeros(3), h)

    q2, s_axis2, h2 = check_screw_parameters(q, 2.0 * np.array(s_axis), h)
    assert_array_almost_equal(q, q2)
    assert_almost_equal(h, h2)
    assert_almost_equal(np.linalg.norm(s_axis2), 1.0)

    q2, s_axis2, h2 = check_screw_parameters(q, s_axis, h)
    assert_array_almost_equal(q, q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert_almost_equal(h, h2)

    q2, s_axis2, h2 = check_screw_parameters(q, s_axis, np.inf)
    assert_array_almost_equal(np.zeros(3), q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert_almost_equal(np.inf, h2)


def test_check_screw_axis():
    random_state = np.random.RandomState(73)
    omega = norm_vector(random_vector(random_state, 3))
    v = random_vector(random_state, 3)

    assert_raises_regexp(
        ValueError, "Expected 3D vector with shape",
        check_screw_axis, np.r_[0, 1, v])

    assert_raises_regexp(
        ValueError, "Norm of rotation axis must either be 0 or 1",
        check_screw_axis, np.r_[2 * omega, v])

    assert_raises_regexp(
        ValueError, "If the norm of the rotation axis is 0",
        check_screw_axis, np.r_[0, 0, 0, v])

    S_pure_translation = np.r_[0, 0, 0, norm_vector(v)]
    S = check_screw_axis(S_pure_translation)
    assert_array_almost_equal(S, S_pure_translation)

    S_both = np.r_[omega, v]
    S = check_screw_axis(S_both)
    assert_array_almost_equal(S, S_both)


def test_check_exponential_coordinates():
    assert_raises_regexp(
        ValueError, "Expected array-like with shape",
        check_exponential_coordinates, [0])

    Stheta = [0.0, 1.0, 2.0, -5.0, -2, 3]
    Stheta2 = check_exponential_coordinates(Stheta)
    assert_array_almost_equal(Stheta, Stheta2)


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


def test_random_screw_axis():
    random_state = np.random.RandomState(893)
    for _ in range(5):
        S = random_screw_axis(random_state)
        check_screw_axis(S)


def test_conversions_between_screw_axis_and_exponential_coordinates():
    S = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    theta = 1.0
    Stheta = exponential_coordinates_from_screw_axis(S, theta)
    S2, theta2 = screw_axis_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(S, S2)
    assert_almost_equal(theta, theta2)

    S = np.zeros(6)
    theta = 0.0
    Stheta = exponential_coordinates_from_screw_axis(S, theta)
    S2, theta2 = screw_axis_from_exponential_coordinates(Stheta)
    assert_array_almost_equal(S, S2)
    assert_almost_equal(theta, theta2)

    random_state = np.random.RandomState(33)
    for _ in range(5):
        S = random_screw_axis(random_state)
        theta = random_state.rand()
        Stheta = exponential_coordinates_from_screw_axis(S, theta)
        S2, theta2 = screw_axis_from_exponential_coordinates(Stheta)
        assert_array_almost_equal(S, S2)
        assert_almost_equal(theta, theta2)


def test_conversions_between_exponential_coordinates_and_transform_log():
    random_state = np.random.RandomState(22)
    for _ in range(5):
        Stheta = random_state.randn(6)
        transform_log = transform_log_from_exponential_coordinates(Stheta)
        Stheta2 = exponential_coordinates_from_transform_log(transform_log)
        assert_array_almost_equal(Stheta, Stheta2)


def test_conversions_between_unit_twist_and_screw_axis():
    random_state = np.random.RandomState(83)
    for _ in range(5):
        S = random_screw_axis(random_state)
        V = unit_twist_from_screw_axis(S)
        S2 = screw_axis_from_unit_twist(V)
        assert_array_almost_equal(S, S2)


def test_conversions_between_unit_twist_and_transform_log():
    V = np.array([[0.0, 0.0, 0.0, 1.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
    theta = 1.0
    transform_log = transform_log_from_unit_twist(V, theta)
    V2, theta2 = unit_twist_from_transform_log(transform_log)
    assert_array_almost_equal(V, V2)
    assert_almost_equal(theta, theta2)

    V = np.zeros((4, 4))
    theta = 0.0
    transform_log = transform_log_from_unit_twist(V, theta)
    V2, theta2 = unit_twist_from_transform_log(transform_log)
    assert_array_almost_equal(V, V2)
    assert_almost_equal(theta, theta2)

    random_state = np.random.RandomState(65)
    for _ in range(5):
        S = random_screw_axis(random_state)
        theta = np.random.rand()
        V = unit_twist_from_screw_axis(S)
        transform_log = transform_log_from_unit_twist(V, theta)
        V2, theta2 = unit_twist_from_transform_log(transform_log)
        assert_array_almost_equal(V, V2)
        assert_almost_equal(theta, theta2)
