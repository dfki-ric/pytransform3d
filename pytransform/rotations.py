import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .plot_utils import Arrow3D
from numpy.testing import assert_array_almost_equal


unitx = np.array([1.0, 0.0, 0.0])
unity = np.array([0.0, 1.0, 0.0])
unitz = np.array([0.0, 0.0, 1.0])


def norm_vector(v):
    """Normalize vector.

    Parameters
    ----------
    v : array-like, shape (n,)
        nd vector

    Returns
    -------
    u : array-like, shape (n,)
        nd unit vector with norm 1 or the zero vector
    """
    norm = np.linalg.norm(v)
    if norm == 0.0:
        return v
    else:
        return v / norm


def norm_axis_angle(a):
    """Normalize axis-angle representation.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The length
        of the axis vector is 1 and the angle is in [0, pi). No rotation
        is represented by [1, 0, 0, 0].
    """
    angle = a[3]
    norm = np.linalg.norm(a[:3])
    if angle % np.pi == 0.0 or norm == 0.0:  # TODO angle % np.pi is not correctly handled
        return np.array([1.0, 0.0, 0.0, 0.0])

    res = np.empty(4)
    res[:3] = a[:3] / norm

    while angle < 0.0:
        angle += 2.0 * np.pi
    while angle >= np.pi:
        angle -= 2.0 * np.pi
    if angle < 0.0:
        angle = np.pi - np.abs(angle)
        res[:3] *= -1

    res[3] = angle

    return res


def perpendicular_to_vectors(a, b):
    """Compute perpendicular vector to two other vectors.

    Parameters
    ----------
    a : array-like, shape (3,)
        3d vector

    b : array-like, shape (3,)
        3d vector

    Returns
    -------
    c : array-like, shape (3,)
        3d vector that is orthogonal to a and b
    """
    return np.cross(a, b)


def angle_between_vectors(a, b, fast=False):
    """Compute angle between to vectors.

    Parameters
    ----------
    a : array-like, shape (n,)
        nd vector

    b : array-like, shape (n,)
        nd vector

    fast : bool, optional (default: False)
        Use fast implementation instead of numerically stable solution

    Returns
    -------
    angle : float
        Angle between a and b
    """
    if len(a) != 3 or fast:
        return np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    else:
        return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))


def random_vector(random_state=np.random.RandomState(0), n=3):
    """Generate an nd vector with normally distributed components.

    Each component will be sampled from :math:`\mathcal{N}(\mu=0, \sigma=1)`.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    n : int, optional (default: 3)
        Number of vector components

    Returns
    -------
    v : array-like, shape (n,)
        Random vector
    """
    return random_state.randn(n)


def random_axis_angle(random_state=np.random.RandomState(0)):
    """Generate random axis-angle.

    The angle will be sampled uniformly from the interval :math:`[0, \pi)`
    and each component of the rotation axis will be sampled from
    :math:`\mathcal{N}(\mu=0, \sigma=1)` and than the axis will be normalized
    to length 1.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)
    """
    angle = np.pi * random_state.rand()
    a = np.array([0, 0, 0, angle])
    a[:3] = norm_vector(random_state.randn(3))
    return a


def random_quaternion(random_state=np.random.RandomState(0)):
    """Generate random quaternion.

    Parameters
    ----------
    random_state : np.random.RandomState, optional (default: random seed 0)
        Random number generator

    Returns
    -------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    return norm_vector(random_state.randn(4))


def cross_product_matrix(v):
    """Generate the cross-product matrix of a vector.

    The cross-product matrix :math:`\\boldsymbol{V}` satisfies the equation

    .. math::

        \\boldsymbol{V} \\boldsymbol{w} = \\boldsymbol{v} \\times
        \\boldsymbol{w}

    Parameters
    ----------
    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    V : array-like, shape (3, 3)
        Cross-product matrix
    """
    return np.array([[  0.0, -v[2],  v[1]],
                     [ v[2],   0.0, -v[0]],
                     [-v[1],  v[0],  0.0]])


def matrix_from_axis_angle(a):
    """Compute rotation matrix from axis-angle.

    This is called exponential map or Rodrigues' formula.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    a = norm_axis_angle(a)
    ux, uy, uz, theta = a
    c = np.cos(theta)
    s = np.sin(theta)
    ci = 1.0 - c
    R = np.array([[ci * ux * ux + c,
                   ci * ux * uy - uz * s,
                   ci * ux * uz + uy * s],
                  [ci * uy * ux + uz * s,
                   ci * uy * uy + c,
                   ci * uy * uz - ux * s],
                  [ci * uz * ux - uy * s,
                   ci * uz * uy + ux * s,
                   ci * uz * uz + c],
                  ])

    # This is equivalent to
    # R = (np.eye(3) * np.cos(theta) +
    #      (1.0 - np.cos(theta)) * a[:3, np.newaxis].dot(a[np.newaxis, :3]) +
    #      cross_product_matrix(a[:3]) * np.sin(theta))

    return R


def matrix_from_quaternion(q):
    """Compute rotation matrix from quaternion.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    uq = norm_vector(q)
    w, x, y, z = uq
    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    R = np.array([[1.0 - y2 - z2,       xy - zw,       xz + yw],
                  [      xy + zw, 1.0 - x2 - z2,       yz - xw],
                  [      xz - yw,       yz + xw, 1.0 - x2 - y2]])
    return R


def matrix_from_angle(basis, angle):
    """Compute rotation matrix from rotation around basis vector.

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angle : float
        Rotation angle

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)

    if basis == 0:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0,   c,   s],
                      [0.0,  -s,   c]])
    elif basis == 1:
        R = np.array([[  c, 0.0,  -s],
                      [0.0, 1.0, 0.0],
                      [  s, 0.0,   c]])
    elif basis == 2:
        R = np.array([[  c,   s, 0.0],
                      [ -s,   c, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R


def matrix_from_euler_xyz(e):
    """Compute rotation matrix from xyz Euler angles.

    The xyz convention is usually used in physics and chemistry.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    alpha, beta, gamma = e
    Qx = matrix_from_angle(0, alpha)
    Qy = matrix_from_angle(1, beta)
    Qz = matrix_from_angle(2, gamma)
    R = Qx.dot(Qy).dot(Qz)
    return R


def matrix_from_euler_zyx(e):
    """Compute rotation matrix from zyx Euler angles.

    The zyx convention is usually used for aircraft dynamics.

    Parameters
    ----------
    e : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    gamma, beta, alpha = e
    Qz = matrix_from_angle(2, gamma)
    Qy = matrix_from_angle(1, beta)
    Qx = matrix_from_angle(0, alpha)
    R = Qz.dot(Qy).dot(Qx)
    return R


def matrix_from(R=None, a=None, q=None, e_xyz=None, e_zyx=None):
    """Compute rotation matrix from another representation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    e_xyz : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes

    e_zyx : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes

    Returns
    -------
    R : array-like, shape (3, 3)
        Rotation matrix
    """
    # TODO test
    if R is not None:
        return R
    if a is not None:
        return matrix_from_axis_angle(a)
    if q is not None:
        return matrix_from_quaternion(q)
    if e_xyz is not None:
        return matrix_from_euler_xyz(e_xyz)
    if e_zyx is not None:
        return matrix_from_euler_zyx(e_zyx)
    raise ValueError("Cannot compute rotation matrix from no rotation.")


def euler_xyz_from_matrix(R):
    """Compute xyz Euler angles from rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    e_xyz : array-like, shape (3,)
        Angles for rotation around x-, y'-, and z''-axes
    """
    if np.abs(R[0, 2]) != 1.0:
        # NOTE: There are two solutions: angle2 and pi - angle2!
        angle2 = np.arcsin(-R[0, 2])
        angle1 = np.arctan2(R[1, 2] / np.cos(angle2), R[2, 2] / np.cos(angle2))
        angle3 = np.arctan2(R[0, 1] / np.cos(angle2), R[0, 0] / np.cos(angle2))
        return np.array([angle1, angle2, angle3])
    else:
        raise NotImplementedError("Gimbal lock! This is note implemented.")
        # TODO implement gimbal lock fix based on
        # http://staff.city.ac.uk/~sbbh653/publications/euler.pdf


def euler_zyx_from_matrix(R):
    """Compute zyx Euler angles from rotation matrix.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    e_zyx : array-like, shape (3,)
        Angles for rotation around z-, y'-, and x''-axes
    """
    if np.abs(R[0, 2]) != 1.0:
        # NOTE: There are two solutions: angle2 and pi - angle2!
        angle2 = np.arcsin(R[2, 0])
        angle3 = np.arctan2(-R[2, 1] / np.cos(angle2), R[2, 2] / np.cos(angle2))
        angle1 = np.arctan2(-R[1, 0] / np.cos(angle2), R[0, 0] / np.cos(angle2))
        return np.array([angle1, angle2, angle3])
    else:
        raise NotImplementedError("Gimbal lock! This is note implemented.")
        # TODO implement gimbal lock fix based on
        # http://staff.city.ac.uk/~sbbh653/publications/euler.pdf


def axis_angle_from_matrix(R):
    """Compute axis-angle from rotation matrix.

    This operation is called logarithmic map.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi].
    """
    angle = np.arccos((np.trace(R) - 1.0) / 2.0)

    if angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    a = np.empty(4)
    if angle == np.pi:
        a[:3] = 0.0
        axis = np.argmax(np.abs(r))
        a[axis] = 1.0
    else:
        a[:3] = r / (2.0 * np.sin(angle))
    a[3] = angle
    return a


def axis_angle_from_quaternion(q):
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    p = q[1:]
    p_norm = np.linalg.norm(p)
    if p_norm < np.finfo(float).eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = p / p_norm
        angle = (2.0 * np.arccos(q[0]),)
        return np.hstack((axis, angle))


def quaternion_from_matrix(R):
    """Compute quaternion from rotation matrix.

    .. warning::

        When computing a quaternion from the rotation matrix there is a sign
        ambiguity: q and -q represent the same rotation.

    Parameters
    ----------
    R : array-like, shape (3, 3)
        Rotation matrix

    Returns
    -------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    q = np.empty(4)
    q[0] = 0.5 * np.sqrt(1.0 + np.trace(R))
    q[1] = 0.25 / q[0] * (R[2, 1] - R[1, 2])
    q[2] = 0.25 / q[0] * (R[0, 2] - R[2, 0])
    q[3] = 0.25 / q[0] * (R[1, 0] - R[0, 1])
    return q


def quaternion_from_axis_angle(a):
    """Compute quaternion from axis-angle.

    This operation is called exponential map.

    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: (x, y, z, angle)

    Returns
    -------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    theta = a[3]
    u = norm_vector(a[:3])

    q = np.empty(4)
    q[0] = np.cos(theta / 2)
    q[1:] = np.sin(theta / 2) * u
    return q


def concatenate_quaternions(q1, q2):
    """Concatenate two quaternions.

    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-like, shape (4,)
        Second quaternion

    Returns
    -------
    q12 : array-like, shape (4,)
        Quaternion that represents the concatenated rotation q1 * q2
    """
    q12 = np.empty(4)
    q12[0] = q1[0] * q2[0] - np.dot(q1[1:], q2[1:])
    q12[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    return q12


def q_prod_vector(q, v):
    """Apply rotation represented by a quaternion to a vector.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    v : array-like, shape (3,)
        3d vector

    Returns
    -------
    w : array-like, shape (3,)
        3d vector
    """
    t = 2 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def q_conj(q):
    """Conjugate of quaternion.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    q_c : array-like, shape (4,)
        Conjugate (w, -x, -y, -z)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def axis_angle_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start axis of rotation and rotation angle: (x, y, z, angle)

    end : array-like, shape (4,)
        Goal axis of rotation and rotation angle: (x, y, z, angle)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    a : array-like, shape (4,)
        Interpolated axis of rotation and rotation angle: (x, y, z, angle)
    """
    omega = angle_between_vectors(start[:3], end[:3])
    w1 = np.sin((1.0 - t) * omega) / np.sin(omega)
    w2 = np.sin(t * omega) / np.sin(omega)
    w1 = np.array([w1, w1, w1, (1.0 - t)])
    w2 = np.array([w2, w2, w2, t])
    return w1 * start + w2 * end


def quaternion_slerp(start, end, t):
    """Spherical linear interpolation.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start unit quaternion to represent rotation: (w, x, y, z)

    end : array-like, shape (4,)
        End unit quaternion to represent rotation: (w, x, y, z)

    t : float in [0, 1]
        Position between start and goal

    Returns
    -------
    q : array-like, shape (4,)
        Interpolated unit quaternion to represent rotation: (w, x, y, z)
    """
    omega = angle_between_vectors(start, end)
    w1 = np.sin((1.0 - t) * omega) / np.sin(omega)
    w2 = np.sin(t * omega) / np.sin(omega)
    return w1 * start + w2 * end


def quaternion_dist(q1, q2):
    """Compute distance between two quaternions.

    We use the angular metric of :math:`S^3`, which is defined as

    .. math::

        d(q_1, q_2) = \\begin{cases}
                      2 || \\log(q_1 * \\overline{q_2})||, \\quad
                        q_1 * \\overline{q_2} \\neq (-1, 0, 0, 0)\\\\
                      2 \\pi, \\text{ otherwise}
                      \\end{cases}


    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion

    q2 : array-like, shape (4,)
        Second quaternion

    Returns
    -------
    dist : float
        Distance between q1 and q2
    """
    q12c = concatenate_quaternions(q1, q_conj(q2))
    if np.any(q12c != np.array([1, 0, 0, 0])):
        return axis_angle_from_quaternion(q12c)[-1]
    else:
        return 2.0 * np.pi


def plot_basis(ax=None, R=np.eye(3), p=np.zeros(3), s=1.0, ax_s=1, **kwargs):
    """Plot basis of a rotation matrix.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    R : array-like, shape (3, 3), optional (default: I)
        Rotation matrix, each column contains a basis vector

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha
    """
    if ax is None:
        ax = _make_new_axis(ax_s)

    for d, c in enumerate(["r", "g", "b"]):
        ax.plot([p[0], p[0] + s * R[0, d]],
                [p[1], p[1] + s * R[1, d]],
                [p[2], p[2] + s * R[2, d]], color=c, lw=3, **kwargs)

    return ax


def plot_axis_angle(ax=None, a=np.array([1, 0, 0, 0]), p=np.zeros(3),
                    s=1.0, ax_s=1, **kwargs):
    """Plot rotation axis and angle.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    a : array-like, shape (4,), optional (default: [1, 0, 0, 0])
        Axis of rotation and rotation angle: (x, y, z, angle)

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Offset from the origin

    s : float, optional (default: 1)
        Scaling of the axis and angle that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha
    """
    if ax is None:
        ax = _make_new_axis(ax_s)

    ax.plot([p[0], p[0] + 0.9 * s * a[0]],
            [p[1], p[1] + 0.9 * s * a[1]],
            [p[2], p[2] + 0.9 * s * a[2]], color="k", lw=3, **kwargs)
    ax.add_artist(Arrow3D([p[0] + s * a[0], p[0] + 1.1 * s * a[0]],
                          [p[1] + s * a[1], p[1] + 1.1 * s * a[1]],
                          [p[2] + s * a[2], p[2] + 1.1 * s * a[2]],
                          mutation_scale=20, lw=1, arrowstyle="-|>", color="k"))

    p1 = (unitx if np.abs(a[0]) <= np.finfo(float).eps else
          perpendicular_to_vectors(unity, a[:3]))
    p2 = perpendicular_to_vectors(a[:3], p1)

    om = angle_between_vectors(p1, p2)
    arc = np.empty((100, 3))
    for i, t in enumerate(np.linspace(0, 2 * a[3] / np.pi, 100)):
        w = np.array([np.sin((1.0 - t) * om), np.sin(t * om)]) / np.sin(om)
        arc[i] = p + 0.5 * s * a[:3] + s * w[0] * p1 + s * w[1] * p2
    ax.plot(arc[:-5, 0], arc[:-5, 1], arc[:-5, 2], color="k", lw=3, **kwargs)
    ax.add_artist(Arrow3D(arc[-2:, 0], arc[-2:, 1], arc[-2:, 2],
                          mutation_scale=20, lw=1, arrowstyle="-|>", color="k"))

    return ax


def _make_new_axis(ax_s):
    ax = plt.subplot(111, projection="3d", aspect="equal")
    ax.set_xlim((-ax_s, ax_s))
    ax.set_ylim((-ax_s, ax_s))
    ax.set_zlim((-ax_s, ax_s))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return ax


def assert_axis_angle_equal(a1, a2, *args, **kwargs):
    """Raise an assertion if two axis-angle are not approximately equal.

    Usually we assume that the rotation axis is normalized to length 1 and
    the angle is within [0, pi). However, this function ignores these
    constraints and will normalize the representations before comparison.
    See numpy.testing.assert_array_almost_equal for a more detailed
    documentation of the other parameters.
    """
    a1 = norm_axis_angle(a1)
    a2 = norm_axis_angle(a2)
    assert_array_almost_equal(a1, a2, *args, **kwargs)


def assert_quaternion_equal(q1, q2, *args, **kwargs):
    """Raise an assertion if two quaternions are not approximately equal.

    Note that quaternions are equal either if q1 == q2 or if q1 == -q2. See
    numpy.testing.assert_array_almost_equal for a more detailed documentation
    of the other parameters.
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
    """
    assert_array_almost_equal(np.dot(R, R.T), np.eye(3))
    assert_array_almost_equal(np.linalg.det(R), 1.0)
