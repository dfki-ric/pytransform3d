"""Transform operations."""

import numpy as np

from ._screws import transform_from_exponential_coordinates
from ._transform import check_transform, exponential_coordinates_from_transform
from ..rotations import (
    axis_angle_from_matrix,
    matrix_from_axis_angle,
    norm_vector,
)


def invert_transform(A2B, strict_check=True, check=True):
    """Invert transform.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrix is valid

    Returns
    -------
    B2A : array, shape (4, 4)
        Transform from frame B to frame A

    See Also
    --------
    pytransform3d.uncertainty.invert_uncertain_transform :
        Invert transformation with uncertainty.

    pytransform3d.trajectories.invert_transforms :
        Invert multiple transformations.
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
    # NOTE there is a faster version, but it is not faster than matrix
    # inversion with numpy:
    # ( R t )^-1   ( R^T -R^T*t )
    # ( 0 1 )    = ( 0    1     )
    return np.linalg.inv(A2B)


def vector_to_point(v):
    """Convert 3D vector to position.

    A point (x, y, z) given by the components of a vector will be represented
    by [x, y, z, 1] in homogeneous coordinates to which we can apply a
    transformation.

    Parameters
    ----------
    v : array-like, shape (3,)
        3D vector that contains x, y, and z

    Returns
    -------
    p : array-like, shape (4,)
        Point vector with 1 as last element

    See Also
    --------
    pytransform3d.transformations.vectors_to_points
        Convert 3D vectors to points in homogeneous coordinates.

    pytransform3d.transformations.vector_to_direction
        Convert one 3D vector to a direction in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_directions
        Convert 3D vectors to directions in homogeneous coordinates.
    """
    return np.hstack((v, 1))


def vectors_to_points(V):
    """Convert 3D vectors to positions.

    A point (x, y, z) given by the components of a vector will be represented
    by [x, y, z, 1] in homogeneous coordinates to which we can apply a
    transformation.

    Parameters
    ----------
    V : array-like, shape (n_points, 3)
        Each row is a 3D vector that contains x, y, and z

    Returns
    -------
    P : array-like, shape (n_points, 4)
        Each row is a point vector with 1 as last element

    See Also
    --------
    pytransform3d.transformations.vector_to_point
        Convert one 3D vector to a point in homogeneous coordinates.

    pytransform3d.transformations.vector_to_direction
        Convert one 3D vector to a direction in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_directions
        Convert 3D vectors to directions in homogeneous coordinates.
    """
    return np.hstack((V, np.ones((len(V), 1))))


def vector_to_direction(v):
    """Convert 3D vector to direction.

    A direction (x, y, z) given by the components of a vector will be
    represented by [x, y, z, 0] in homogeneous coordinates to which we can
    apply a transformation.

    Parameters
    ----------
    v : array-like, shape (3,)
        3D vector that contains x, y, and z

    Returns
    -------
    p : array-like, shape (4,)
        Direction vector with 0 as last element

    See Also
    --------
    pytransform3d.transformations.vector_to_point
        Convert one 3D vector to a point in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_points
        Convert 3D vectors to points in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_directions
        Convert 3D vectors to directions in homogeneous coordinates.
    """
    return np.hstack((v, 0))


def vectors_to_directions(V):
    """Convert 3D vectors to directions.

    A direction (x, y, z) given by the components of a vector will be
    represented by [x, y, z, 0] in homogeneous coordinates to which we can
    apply a transformation.

    Parameters
    ----------
    V : array-like, shape (n_directions, 3)
        Each row is a 3D vector that contains x, y, and z

    Returns
    -------
    P : array-like, shape (n_directions, 4)
        Each row is a direction vector with 0 as last element

    See Also
    --------
    pytransform3d.transformations.vector_to_point
        Convert one 3D vector to a point in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_points
        Convert 3D vectors to points in homogeneous coordinates.

    pytransform3d.transformations.vector_to_direction
        Convert one 3D vector to a direction in homogeneous coordinates.
    """
    return np.hstack((V, np.zeros((len(V), 1))))


def concat(A2B, B2C, strict_check=True, check=True):
    r"""Concatenate transformations.

    We use the extrinsic convention, which means that B2C is left-multiplied
    to A2B. In mathematical notation:

    .. math::

        \boldsymbol{T}_{CB} \boldsymbol{T}_{BA} = \boldsymbol{T}_{CA}.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    B2C : array-like, shape (4, 4)
        Transform from frame B to frame C

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrices are valid

    Returns
    -------
    A2C : array-like, shape (4, 4)
        Transform from frame A to frame C

    See Also
    --------
    pytransform3d.trajectories.concat_one_to_many :
        Concatenate one transformation with multiple transformations.

    pytransform3d.trajectories.concat_many_to_one :
        Concatenate multiple transformations with one.

    pytransform3d.uncertainty.concat_globally_uncertain_transforms :
        Concatenate two independent globally uncertain transformations.

    pytransform3d.uncertainty.concat_locally_uncertain_transforms :
        Concatenate two independent locally uncertain transformations.
    """
    if check:
        A2B = check_transform(A2B, strict_check=strict_check)
        B2C = check_transform(B2C, strict_check=strict_check)
    return B2C.dot(A2B)


def transform(A2B, PA, strict_check=True):
    """Transform point or list of points or directions.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    PA : array-like, shape (4,) or (n_points, 4)
        One of multiple points or directions in frame A that are represented
        by homogeneous coordinates.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    PB : array-like, shape (4,) or (n_points, 4)
        One of multiple points or directions in frame B that are represented
        by homogeneous coordinates.

    Raises
    ------
    ValueError
        If dimensions are incorrect

    See Also
    --------
    pytransform3d.transformations.vector_to_point
        Convert one 3D vector to a point in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_points
        Convert 3D vectors to points in homogeneous coordinates.

    pytransform3d.transformations.vector_to_direction
        Convert one 3D vector to a direction in homogeneous coordinates.

    pytransform3d.transformations.vectors_to_directions
        Convert 3D vectors to directions in homogeneous coordinates.
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    PA = np.asarray(PA)

    if PA.ndim == 1:
        return np.dot(A2B, PA)

    if PA.ndim == 2:
        return np.dot(PA, A2B.T)

    raise ValueError("Cannot transform array with more than 2 dimensions")


def scale_transform(
    A2B,
    s_xr=1.0,
    s_yr=1.0,
    s_zr=1.0,
    s_r=1.0,
    s_xt=1.0,
    s_yt=1.0,
    s_zt=1.0,
    s_t=1.0,
    s_d=1.0,
    strict_check=True,
):
    """Scale a transform from A to reference frame B.

    See algorithm 10 from "Analytic Approaches for Design and Operation of
    Haptic Human-Machine Interfaces" (Bertold Bongardt).

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    s_xr : float, optional (default: 1)
        Scaling of x-component of the rotation axis

    s_yr : float, optional (default: 1)
        Scaling of y-component of the rotation axis

    s_zr : float, optional (default: 1)
        Scaling of z-component of the rotation axis

    s_r : float, optional (default: 1)
        Scaling of the rotation

    s_xt : float, optional (default: 1)
        Scaling of z-component of the translation

    s_yt : float, optional (default: 1)
        Scaling of z-component of the translation

    s_zt : float, optional (default: 1)
        Scaling of z-component of the translation

    s_t : float, optional (default: 1)
        Scaling of the translation

    s_d : float, optional (default: 1)
        Scaling of the whole transform (displacement)

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    Returns
    -------
    A2B_scaled
        Scaled transform from frame A to frame B (actually this is a transform
        from A to another frame C)
    """
    A2B = check_transform(A2B, strict_check=strict_check)
    A2B_scaled = np.eye(4)

    R = A2B[:3, :3]
    t = A2B[:3, 3]

    S_t = np.array([s_xt, s_yt, s_zt])
    A2B_scaled[:3, 3] = s_d * s_t * S_t * t

    a = axis_angle_from_matrix(R)
    a_new = np.empty(4)
    a_new[3] = s_d * s_r * a[3]
    S_r = np.array([s_xr, s_yr, s_zr])
    a_new[:3] = norm_vector(S_r * a[:3])
    A2B_scaled[:3, :3] = matrix_from_axis_angle(a_new)

    return A2B_scaled


def transform_sclerp(start, end, t):
    r"""Screw linear interpolation (ScLERP) for transformation matrices.

    We compute the difference between two poses
    :math:`\boldsymbol{T}_{AB} = \boldsymbol{T}^{-1}_{WA}\boldsymbol{T}_{WB}`,
    convert it to exponential coordinates
    :math:`Log(\boldsymbol{T}_{AB}) = \mathcal{S}\theta_{AB}`, compute a
    fraction of it :math:`t\mathcal{S}\theta_{AB}` with :math:`t \in [0, 1]`,
    and use the exponential map to concatenate the fraction of the difference
    :math:`\boldsymbol{T}_{WA} Exp(t\mathcal{S}\theta_{AB})`.

    Parameters
    ----------
    start : array-like, shape (4, 4)
        Transformation matrix to represent start pose.

    end : array-like, shape (4, 4)
        Transformation matrix to represent end pose.

    t : float in [0, 1]
        Position between start and goal.

    Returns
    -------
    A2B_t : array, shape (4, 4)
        Interpolated pose.

    See Also
    --------
    dual_quaternion_sclerp :
        ScLERP for dual quaternions.

    pq_slerp :
        An alternative approach is spherical linear interpolation (SLERP) with
        position and quaternion.
    """
    end2start = np.dot(invert_transform(start), end)
    return np.dot(start, transform_power(end2start, t))


def transform_power(A2B, t):
    r"""Compute power of a transformation matrix with respect to scalar.

    .. math::

        \boldsymbol{T}_{BA}^t

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B.

    t : float
        Exponent.

    Returns
    -------
    A2B_t : array, shape (4, 4)
        Transformation matrix.
    """
    Stheta = exponential_coordinates_from_transform(A2B)
    return transform_from_exponential_coordinates(Stheta * t)
