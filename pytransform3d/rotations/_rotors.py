import numpy as np
from ._utils import norm_vector
from ._constants import unitx, unity, unitz
from ._quaternion_operations import concatenate_quaternions, q_prod_vector


def wedge(a, b):  # TODO sphinx
    r"""Outer product of two vectors (also exterior or wedge product).

    .. math::

        B = a \wedge b

    Parameters
    ----------
    a : array-like, shape (3,)
        Vector: (x, y, z)

    b : array-like, shape (3,)
        Vector: (x, y, z)

    Returns
    -------
    B : array, shape (3,)
        Bivector that defines the plane that a and b form together:
        (b_yz, b_zx, b_xy)
    """
    # TODO check inputs
    return np.cross(a, b)


def plane_normal_from_bivector(B):  # TODO sphinx
    """Convert bivector to normal vector of a plane.

    Parameters
    ----------
    B : array-like, shape (3,)
        Bivector that defines a plane: (b_yz, b_zx, b_xy)

    Returns
    -------
    n : array, shape (3,)
        Unit normal of the corresponding plane: (x, y, z)
    """
    # TODO check input
    return norm_vector(B)


def geometric_product(a, b):  # TODO sphinx
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
        Vector: (x, y, z)

    b : array-like, shape (3,)
        Vector: (x, y, z)

    Returns
    -------
    ab : array, shape (4,)
        A multivector (a, b_yz, b_zx, b_xy) composed of scalar and bivector
        (b_yz, b_zx, b_xy) that form the geometric product of vectors a and b.
    """
    # TODO check inputs
    return np.hstack(((np.dot(a, b),), wedge(a, b)))


def rotor_reverse(rotor):  # TODO sphinx
    """Invert rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    reverse_rotor : array, shape (4,)
        Reverse of the rotor: (a, b_yz, b_zx, b_xy)
    """
    # TODO check input
    return np.hstack(((rotor[0],), -rotor[1:]))


def concatenate_rotors(rotor1, rotor2):  # TODO sphinx
    """Concatenate rotors.

    # TODO order of rotation

    Parameters
    ----------
    rotor1 : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    rotor2 : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    rotor : array, shape (4,)
        rotor1 applied to rotor2: (a, b_yz, b_zx, b_xy)
    """
    return concatenate_quaternions(rotor1, rotor2)


def rotor_apply(rotor, v):  # TODO test, sphinx
    r"""Compute rotation matrix from rotor.

    .. math::

        v' = R v R^*

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    v : array-like, shape (3,)
        Vector: (x, y, z)

    Returns
    -------
    v : array, shape (3,)
        Rotated vector
    """
    return q_prod_vector(rotor, v)


def matrix_from_rotor(rotor):  # TODO test, sphinx
    """Compute rotation matrix from rotor.

    Parameters
    ----------
    rotor : array-like, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)

    Returns
    -------
    R : array, shape (3, 3)
        Rotation matrix
    """
    # TODO check input
    return np.column_stack((
        rotor_apply(rotor, unitx), rotor_apply(rotor, unity),
        rotor_apply(rotor, unitz)))


def rotor_from_two_vectors(v_from, v_to):  # TODO test, sphinx
    """Construct the rotor that rotates one vector to another.

    Parameters
    ----------
    v_from : array-like, shape (3,)
        Unit vector (will be normalized internally)

    v_to : array-like, shape (3,)
        Unit vector (will be normalized internally)

    Returns
    -------
    rotor : array, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)
    """
    # TODO check input
    v_from = norm_vector(v_from)
    v_to = norm_vector(v_to)
    return norm_vector(np.hstack(
        ((1.0 + np.dot(v_from, v_to),), wedge(v_from, v_to))))


def rotor_from_plane_angle(B, angle):  # TODO sphinx
    r"""Compute rotor from plane bivector and angle.

    Parameters
    ----------
    B : array-like, shape (3,)
        Unit bivector (b_yz, b_zx, b_xy) that represents the plane of rotation
        (will be normalized internally)

    angle : float
        Rotation angle

    Returns
    -------
    rotor : array, shape (4,)
        Rotor: (a, b_yz, b_zx, b_xy)
    """
    # TODO check input
    a = np.cos(angle / 2.0)
    sina = np.sin(angle / 2.0)
    B = norm_vector(B)
    return np.hstack(((a,), sina * B))
