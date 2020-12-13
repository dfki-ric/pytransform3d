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
                                           check_screw_axis)
from pytransform3d.rotations import (matrix_from, random_axis_angle,
                                     random_vector, axis_angle_from_matrix,
                                     norm_vector)
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

    q2, s_axis2, h2 = check_screw_parameters(q, 2.0 * np.array(s_axis), h)
    assert_array_almost_equal(q, q2)
    assert_almost_equal(h, h2)
    assert_almost_equal(np.linalg.norm(s_axis2), 1.0)

    q2, s_axis2, h2 = check_screw_parameters(q, s_axis, h)
    assert_array_almost_equal(q, q2)
    assert_array_almost_equal(s_axis, s_axis2)
    assert_almost_equal(h, h2)


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
