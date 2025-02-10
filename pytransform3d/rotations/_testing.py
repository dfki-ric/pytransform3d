"""Testing utilities."""
import numpy as np
from numpy.testing import assert_array_almost_equal
from ._axis_angle import norm_axis_angle, norm_compact_axis_angle
from ._euler import norm_euler
from ._mrp import mrp_double


def assert_euler_equal(e1, e2, i, j, k, *args, **kwargs):
    """Raise an assertion if two Euler angles are not approximately equal.

    Parameters
    ----------
    e1 : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    e2 : array-like, shape (3,)
        Rotation angles in radians about the axes i, j, k in this order.

    i : int from [0, 1, 2]
        The first rotation axis (0: x, 1: y, 2: z)

    j : int from [0, 1, 2]
        The second rotation axis (0: x, 1: y, 2: z)

    k : int from [0, 1, 2]
        The third rotation axis (0: x, 1: y, 2: z)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    e1 = norm_euler(e1, i, j, k)
    e2 = norm_euler(e2, i, j, k)
    assert_array_almost_equal(e1, e2, *args, **kwargs)


def assert_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the rotation axis is normalized to length 1 and
    the angle is within [0, pi). However, this function ignores these
    constraints and will normalize the representations before comparison.
    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    a1 : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    a2 : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    a1 = norm_axis_angle(a1)
    a2 = norm_axis_angle(a2)
    # required despite normalization in case of 180 degree rotation
    if np.any(np.sign(a1) != np.sign(a2)):
        a1 = -a1
        a1 = norm_axis_angle(a1)
    assert_array_almost_equal(a1, a2, *args, **kwargs)


def assert_compact_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the angle is within [0, pi). However, this function
    ignores this constraint and will normalize the representations before
    comparison. See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    a1 : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    a2 : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    angle1 = np.linalg.norm(a1)
    angle2 = np.linalg.norm(a2)
    # required despite normalization in case of 180 degree rotation
    if (abs(angle1) == np.pi and abs(angle2) == np.pi and
            any(np.sign(a1) != np.sign(a2))):
        a1 = -a1
    a1 = norm_compact_axis_angle(a1)
    a2 = norm_compact_axis_angle(a2)
    assert_array_almost_equal(a1, a2, *args, **kwargs)


def assert_quaternion_equal(q1, q2, *args, **kwargs):
    """Raise an assertion if two quaternions are not approximately equal.

    Note that quaternions are equal either if q1 == q2 or if q1 == -q2. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.

    Parameters
    ----------
    q1 : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    q2 : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    try:
        assert_array_almost_equal(q1, q2, *args, **kwargs)
    except AssertionError:
        assert_array_almost_equal(q1, -q2, *args, **kwargs)


def assert_rotation_matrix(R, *args, **kwargs):
    """Raise an assertion if a matrix is not a rotation matrix.

    The two properties :math:`\\boldsymbol{I} = \\boldsymbol{R R}^T` and
    :math:`det(R) = 1` will be checked. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    assert_array_almost_equal(np.dot(R, R.T), np.eye(3), *args, **kwargs)
    assert_array_almost_equal(np.linalg.det(R), 1.0, *args, **kwargs)


def assert_mrp_equal(mrp1, mrp2, *args, **kwargs):
    """Raise an assertion if two MRPs are not approximately equal.

    There are two MRPs that represent the same orientation (double cover). See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.

    Parameters
    ----------
    mrp1 : array-like, shape (3,)
        Modified Rodrigues parameters.

    mrp1 : array-like, shape (3,)
        Modified Rodrigues parameters.

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    try:
        assert_array_almost_equal(mrp1, mrp2, *args, **kwargs)
    except AssertionError:
        assert_array_almost_equal(mrp1, mrp_double(mrp2), *args, **kwargs)
