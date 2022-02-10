"""Rotor operations."""
import numpy as np
from ._utils import norm_vector, check_rotor, perpendicular_to_vector
from ._constants import unitx, unity, unitz, eps
from ._quaternion_operations import concatenate_quaternions, q_prod_vector


def wedge(a, b):
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
    return np.cross(a, b)


def plane_normal_from_bivector(B):
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
    return norm_vector(B)


def geometric_product(a, b):
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
    return np.hstack(((np.dot(a, b),), wedge(a, b)))


def rotor_reverse(rotor):
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
    rotor = check_rotor(rotor)
    return np.hstack(((rotor[0],), -rotor[1:]))


def concatenate_rotors(rotor1, rotor2):
    """Concatenate rotors.

    Suppose we want to apply two extrinsic rotations given by rotors
    R1 and R2 to a vector v. We can either apply R2 to v and then R1 to
    the result or we can concatenate R1 and R2 and apply the result to v.

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
    rotor1 = check_rotor(rotor1)
    rotor2 = check_rotor(rotor2)
    return concatenate_quaternions(rotor1, rotor2)


def rotor_apply(rotor, v):
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
    rotor = check_rotor(rotor)
    return q_prod_vector(rotor, v)


def matrix_from_rotor(rotor):
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
    rotor = check_rotor(rotor)
    return np.column_stack((
        rotor_apply(rotor, unitx), rotor_apply(rotor, unity),
        rotor_apply(rotor, unitz)))


def rotor_from_two_directions(v_from, v_to):
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
    v_from = norm_vector(v_from)
    v_to = norm_vector(v_to)
    cos_angle_p1 = 1.0 + np.dot(v_from, v_to)
    if cos_angle_p1 < eps:
        # There is an infinite number of solutions for the plane of rotation.
        # This solution works with our convention, since the rotation axis is
        # the same as the plane bivector.
        plane = perpendicular_to_vector(v_from)
    else:
        plane = wedge(v_from, v_to)
    multivector = np.hstack(((cos_angle_p1,), plane))
    return norm_vector(multivector)


def rotor_from_plane_angle(B, angle):
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
    a = np.cos(angle / 2.0)
    sina = np.sin(angle / 2.0)
    B = norm_vector(B)
    return np.hstack(((a,), sina * B))
