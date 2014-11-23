import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.testing import assert_array_almost_equal


unitx = np.array([1.0, 0.0, 0.0])
unity = np.array([0.0, 1.0, 0.0])
unitz = np.array([0.0, 0.0, 1.0])


def norm_vector(v):
    """Normalize vector."""
    return v / np.linalg.norm(v)


def random_axis_angle(random_state=np.random.RandomState(0)):
    angle = np.pi * random_state.rand()
    a = np.array([0, 0, 0, angle])
    a[:3] = norm_vector(random_state.rand(3))
    return a


def random_quaternion(random_state=np.random.RandomState(0)):
    return norm_vector(random_state.rand(4))


def perpendicular_to_vectors(a, b):
    return np.cross(a, b)


def angle_between_vectors(a, b):
    # Numerically stable:
    return np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))
    # Faster:
    #cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    #return np.arccos(cos)


def cross_product_matrix(v):
    return np.array([[  0.0, -v[2],  v[1]],
                     [ v[2],   0.0, -v[0]],
                     [-v[1],  v[0],  0.0]])


def matrix_from_axis_angle(a):
    """Exponential map or Rodrigues' formula."""
    e = norm_vector(a[:3])
    theta = a[3]

    R = (np.eye(3) * np.cos(theta) +
         (1.0 - np.cos(theta)) * e[:, np.newaxis].dot(e[np.newaxis, :]) +
         cross_product_matrix(e) * np.sin(theta))

    #"""
    ux, uy, uz = e
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
    #"""
    return R


def matrix_from_quaternion(q):
    uq = q / np.linalg.norm(q)
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
    c = np.cos(angle)
    s = np.sin(angle)

    if basis == 0:
        R = np.array([[1.0, 0.0, 0.0],
                      [0.0,   c,  -s],
                      [0.0,   s,   c]])
    elif basis == 1:
        R = np.array([[  c, 0.0,   s],
                      [0.0, 1.0, 0.0],
                      [ -s, 0.0,   c]])
    elif basis == 2:
        R = np.array([[  c,  -s, 0.0],
                      [  s,   c, 0.0],
                      [0.0, 0.0, 1.0]])
    else:
        raise ValueError("Basis must be in [0, 1, 2]")

    return R


def matrix_from_euler_xyz(e):
    """

    The xyz convention is usually used in physics and chemistry.
    """
    alpha, beta, gamma = e
    Qx = matrix_from_angle(0, alpha)
    Qy = matrix_from_angle(1, beta)
    Qz = matrix_from_angle(2, gamma)
    R = Qx.dot(Qy).dot(Qz)
    return R


def matrix_from_euler_zyx(e):
    """

    The zyx convention is usually used for aircraft dynamics.
    """
    gamma, beta, alpha = e
    Qz = matrix_from_angle(2, gamma)
    Qy = matrix_from_angle(1, beta)
    Qx = matrix_from_angle(0, alpha)
    R = Qz.dot(Qy).dot(Qx)
    return R


def axis_angle_from_matrix(R):
    """Logarithmic map.

    The angle of the axis-angle representation is constrained to [0, pi) so
    that the mapping is unique.
    """
    if np.all(R == np.eye(3)):
        return np.zeros(4)
    else:
        a = np.empty(4)
        a[3] = np.arccos((np.trace(R) - 1.0) / 2.0)
        r = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        a[:3] = r / (2.0 * np.sin(a[3]))
        if a[3] >= np.pi:
            raise Exception("Angle must be within [0, pi) but is %g" % a[3])
        return a


def axis_angle_from_quaternion(q):
    p = q[1:]
    p_norm = np.linalg.norm(p)
    if p_norm < np.finfo(float).eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        axis = p / p_norm
        angle = (2.0 * np.arccos(q[0]),)
        return np.hstack((axis, angle))


def axis_angle_slerp(start, end, t):
    omega = angle_between_vectors(start[:3], end[:3])
    w1 = np.sin((1.0 - t) * omega) / np.sin(omega)
    w2 = np.sin(t * omega) / np.sin(omega)
    w1 = np.array([w1, w1, w1, (1.0 - t)])
    w2 = np.array([w2, w2, w2, t])
    return w1 * start + w2 * end


def quaternion_from_matrix(R):
    """

    When computing a quaternion from the rotation matrix there is a sign ambiguity: q and -q represent the same rotation.
    """
    q = np.empty(4)
    q[0] = 0.5 * np.sqrt(1.0 + np.trace(R))
    q[1] = 0.25 / q[0] * (R[2, 1] - R[1, 2])
    q[2] = 0.25 / q[0] * (R[0, 2] - R[2, 0])
    q[3] = 0.25 / q[0] * (R[1, 0] - R[0, 1])
    return q


def quaternion_from_axis_angle(a):
    ua = a.copy()
    ua[:3] /= np.linalg.norm(ua)

    theta = ua[3]
    q = np.empty(4)
    q[0] = np.cos(theta / 2)
    q[1:] = np.sin(theta / 2) * ua[:3]
    return q


def q_conj(q):
    conj = q.copy()
    conj[1:] *= -1
    return coj


def q_prod(q1, q2):
    return np.array([q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
                     q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] + q1[3] * q2[2],
                     q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
                     q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]])


def check_rotation_matrix(R):
    assert_array_almost_equal(np.dot(R, R.T), np.eye(3))
    assert_array_almost_equal(np.linalg.det(R), 1.0)


def plot_basis(ax=None, R=np.eye(3), p=np.zeros(3), s=1.0, ax_s=1,
               **kwargs):
    if ax is None:
        ax = plt.subplot(111, projection="3d", aspect="equal")
        ax.set_xlim((-ax_s, ax_s))
        ax.set_ylim((-ax_s, ax_s))
        ax.set_zlim((-ax_s, ax_s))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    ax.plot([p[0], p[0] + s * R[0, 0]],
            [p[1], p[1] + s * R[1, 0]],
            [p[2], p[2] + s * R[2, 0]], color="r", lw=3, **kwargs)
    ax.plot([p[0], p[0] + s * R[0, 1]],
            [p[1], p[1] + s * R[1, 1]],
            [p[2], p[2] + s * R[2, 1]], color="g", lw=3, **kwargs)
    ax.plot([p[0], p[0] + s * R[0, 2]],
            [p[1], p[1] + s * R[1, 2]],
            [p[2], p[2] + s * R[2, 2]], color="b", lw=3, **kwargs)

    return ax


def plot_axis_angle(ax=None, a=np.array([1, 0, 0, 0]), p=np.zeros(3),
                    s=1.0, ax_s=1, **kwargs):
    if ax is None:
        ax = plt.subplot(111, projection="3d", aspect="equal")
        ax.set_xlim((-ax_s, ax_s))
        ax.set_ylim((-ax_s, ax_s))
        ax.set_zlim((-ax_s, ax_s))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    ax.plot([p[0], p[0] + s * a[0]],
            [p[1], p[1] + s * a[1]],
            [p[2], p[2] + s * a[2]], color="k", lw=3, **kwargs)

    if np.abs(a[0]) <= np.finfo(float).eps:
        p1 = unitx
    else:
        p1 = perpendicular_to_vectors(unity, a[:3])

    p2 = perpendicular_to_vectors(a[:3], p1)
    omega = angle_between_vectors(p1, p2)
    arc = np.empty((100, 3))
    for i, t in enumerate(np.linspace(0, 2 * a[3] / np.pi, 100)):
        w1 = np.sin((1.0 - t) * omega) / np.sin(omega)
        w2 = np.sin(t * omega) / np.sin(omega)
        arc[i] = p + 0.5 * s * a[:3] + s * w1 * p1 + s * w2 * p2
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="k", lw=3, **kwargs)
    ax.scatter(arc[0, 0], arc[0, 1], arc[0, 2], color="k", lw=3, **kwargs)

    return ax
