"""Testing utilities."""
import numpy as np
from numpy.testing import assert_array_almost_equal
from ..rotations import assert_rotation_matrix, norm_angle
from ._dual_quaternion_operations import (
    dq_q_conj, concatenate_dual_quaternions)


def assert_transform(A2B, *args, **kwargs):
    """Raise an assertion if the transform is not a homogeneous matrix.

    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    assert_rotation_matrix(A2B[:3, :3], *args, **kwargs)
    assert_array_almost_equal(A2B[3], np.array([0.0, 0.0, 0.0, 1.0]),
                              *args, **kwargs)


def assert_unit_dual_quaternion(dq, *args, **kwargs):
    """Raise an assertion if the dual quaternion does not have unit norm.

    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.

    Parameters
    ----------
    dq : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    real = dq[:4]
    dual = dq[4:]

    real_norm = np.linalg.norm(real)
    assert_array_almost_equal(real_norm, 1.0, *args, **kwargs)

    real_dual_dot = np.dot(real, dual)
    assert_array_almost_equal(real_dual_dot, 0.0, *args, **kwargs)

    # The two previous checks are consequences of the unit norm requirement.
    # The norm of a dual quaternion is defined as the product of a dual
    # quaternion and its quaternion conjugate.
    dq_conj = dq_q_conj(dq)
    dq_prod_dq_conj = concatenate_dual_quaternions(dq, dq_conj)
    assert_array_almost_equal(dq_prod_dq_conj, [1, 0, 0, 0, 0, 0, 0, 0],
                              *args, **kwargs)


def assert_unit_dual_quaternion_equal(dq1, dq2, *args, **kwargs):
    """Raise an assertion if unit dual quaternions are not approximately equal.

    Note that unit dual quaternions are equal either if dq1 == dq2 or if
    dq1 == -dq2. See numpy.testing.assert_array_almost_equal for a more
    detailed documentation of the other parameters.

    Parameters
    ----------
    dq1 : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    dq2 : array-like, shape (8,)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    try:
        assert_array_almost_equal(dq1, dq2, *args, **kwargs)
    except AssertionError:
        assert_array_almost_equal(dq1, -dq2, *args, **kwargs)


def assert_screw_parameters_equal(
        q1, s_axis1, h1, theta1, q2, s_axis2, h2, theta2, *args, **kwargs):
    """Raise an assertion if two sets of screw parameters are not similar.

    Note that the screw axis can be inverted. In this case theta and h have
    to be adapted.

    This function needs the dependency nose.

    Parameters
    ----------
    q1 : array, shape (3,)
        Vector to a point on the screw axis that is orthogonal to s_axis

    s_axis1 : array, shape (3,)
        Unit direction vector of the screw axis

    h1 : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta1 : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    q2 : array, shape (3,)
        Vector to a point on the screw axis that is orthogonal to s_axis

    s_axis2 : array, shape (3,)
        Unit direction vector of the screw axis

    h2 : float
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    theta2 : float
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    args : tuple
        Positional arguments that will be passed to
        `assert_array_almost_equal`

    kwargs : dict
        Positional arguments that will be passed to
        `assert_array_almost_equal`
    """
    from nose.tools import assert_almost_equal

    # normalize thetas
    theta1_new = norm_angle(theta1)
    h1 *= theta1 / theta1_new
    theta1 = theta1_new

    theta2_new = norm_angle(theta2)
    h2 *= theta2 / theta2_new
    theta2 = theta2_new

    # q1 and q2 can be any points on the screw axis, that is, they must be a
    # linear combination of each other and the screw axis (which one does not
    # matter since they should be identical or mirrored)
    q1_to_q2 = q2 - q1
    factors = q1_to_q2 / s_axis2
    assert_almost_equal(factors[0], factors[1])
    assert_almost_equal(factors[1], factors[2])
    try:
        assert_array_almost_equal(s_axis1, s_axis2, *args, **kwargs)
        assert_almost_equal(h1, h2)
        assert_almost_equal(theta1, theta2)
    except AssertionError:  # possibly mirrored screw axis
        s_axis1_new = -s_axis1
        # make sure that we keep the direction of rotation
        theta1_new = 2.0 * np.pi - theta1
        # adjust pitch: switch sign and update rotation component
        h1 = -h1 / theta1_new * theta1
        theta1 = theta1_new

        # we have to normalize the angle again
        theta1_new = norm_angle(theta1)
        h1 *= theta1 / theta1_new
        theta1 = theta1_new

        assert_array_almost_equal(s_axis1_new, s_axis2, *args, **kwargs)
        assert_almost_equal(h1, h2)
        assert_almost_equal(theta1, theta2)
