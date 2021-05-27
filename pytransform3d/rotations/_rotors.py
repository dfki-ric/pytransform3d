import numpy as np
from ._utils import norm_vector
from ._constants import unitx, unity, unitz


def wedge(a, b):  # TODO type hints, sphinx
    r"""Outer product of two vectors (also exterior or wedge product).

    .. math::

        B = a \wedge b

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector

    b : array-like, shape (3,)
        Vector

    Returns
    -------
    B : array, shape (3,)
        Bivector that defines the plane that a and b form together
    """
    # TODO check inputs
    return np.array([
        a[0] * b[1] - a[1] * b[0],  # x wedge y
        a[0] * b[2] - a[2] * b[0],  # x wedge z
        a[1] * b[2] - a[2] * b[1]   # y wedge z
    ])


def geometric_product(a, b):  # TODO type hints, sphinx
    r"""Geometric product of two vectors.

    The geometric product consists of the symmetric inner / dot product and the
    antisymmetric outer product of two vectors.

    .. math::

        ab = a \cdot b + a \wedge b

    The inner product contains the cosine and the outer product contains the
    sine of the angle of rotation from a to b.

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector

    b : array-like, shape (3,)
        Vector

    Returns
    -------
    ab : array, shape (4,)
        A scalar and a bivector that are the geometric product of a and b.
    """
    # TODO check inputs
    return np.hstack(((np.dot(a, b),), wedge(a, b)))


def rotor_reverse(rotor):  # TODO test, type hints, sphinx
    """Invert rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor

    Returns
    -------
    reverse_rotor : array, shape (4,)
        Reverse of the rotor
    """
    # TODO check input
    return np.hstack(((rotor[0],), -rotor[1:]))


def norm_rotor(rotor):  # TODO test, type hints, sphinx
    """Normalize rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor

    Returns
    -------
    norm_rotor : array, shape (4,)
        Normalized rotor
    """
    # TODO check input
    return norm_vector(rotor)


def matrix_from_rotor(rotor):  # TODO test, type hints, sphinx, move to conversions
    """Compute rotation matrix from rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    # TODO check input
    return np.column_stack((
        rotor_apply(rotor, unitx), rotor_apply(rotor, unity),
        rotor_apply(rotor, unitz)))


def concatenate_rotors(rotor1, rotor2):  # TODO test, type hints, sphinx, move to conversions
    """Concatenate rotors.

    # TODO order of rotation

    Parameters
    ----------
    rotor1 : array-like, shape (4,)
        Rotor

    rotor2 : array-like, shape (4,)
        Rotor

    Returns
    -------
    rotor : array, shape (4,)
        rotor1 applied to rotor2
    """
    # TODO check input
    result = np.empty(4)

    result[0] = rotor1[0] * rotor2[0] - rotor1[1] * rotor2[1] - rotor1[2] * rotor2[2] - rotor1[3] * rotor2[3]
    result[1] = rotor1[1] * rotor2[0] + rotor1[0] * rotor2[1] + rotor1[3] * rotor2[2] - rotor1[2] * rotor2[3]
    result[2] = rotor1[2] * rotor2[0] + rotor1[0] * rotor2[2] - rotor1[3] * rotor2[1] + rotor1[1] * rotor2[3]
    result[3] = rotor1[3] * rotor2[0] + rotor1[0] * rotor2[3] + rotor1[2] * rotor2[1] - rotor1[1] * rotor2[2]

    return result


def rotor_apply(rotor, v):  # TODO test, type hints, sphinx
    r"""Compute rotation matrix from rotor.

    .. math::

        v' = R v R^*

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor

    v : array-like, shape (3,)
        Vector

    Returns
    -------
    v : array, shape (3,)
        Rotated vector
    """
    # TODO check input
    a = rotor[0]
    b01 = rotor[1]
    b02 = rotor[2]
    b12 = rotor[3]

    # q = P x
    q = np.empty(3)
    q[0] = a * v[0] + v[1] * b01 + v[2] * b02
    q[1] = a * v[1] - v[0] * b01 + v[2] * b12
    q[2] = a * v[2] - v[0] * b02 - v[1] * b12

    q012 = v[0] * b12 - v[1] * b02 + v[2] * b01

    # r = q P *
    r = np.empty(3)
    r[0] = a * q[0] + q[1] * b01 + q[2] * b02 + q012 * b12
    r[1] = a * q[1] - q[0] * b01 - q012 * b02 + q[2] * b12
    r[2] = a * q[2] + q012 * b01 - q[0] * b02 - q[1] * b12

    return r


def rotor_from_two_vectors(v_from, v_to):  # TODO test, type hints, sphinx, move to conversions
    r"""Compute rotor from two vectors.

    Parameters
    ----------
    v_from : array-like, shape (3,)
        Vector

    v_to : array-like, shape (3,)
        Vector

    Returns
    -------
    rotor : array, shape (4,)
        Rotor
    """
    # TODO check input
    a = 1.0 + np.dot(v_from, v_to)
    minus = wedge(v_to, v_from)
    return norm_vector(np.hstack(((a,), minus)))


def rotor_from_plane_angle(p):  # TODO test, type hints, sphinx, move to conversions
    r"""Compute rotor from plane bivector and angle.

    Parameters
    ----------
    p : array-like, shape (4,)
        Plane of rotation (bivector) and rotation angle:
        (b_xy, b_xz, b_yz, angle)

    Returns
    -------
    rotor : array, shape (4,)
        Rotor
    """
    # TODO check input
    plane = p[:3]
    angle = p[3]
    a = np.cos(angle / 2.0)
    sina = np.sin(angle / 2.0)
    return np.hstack(((a,), -sina * plane))
