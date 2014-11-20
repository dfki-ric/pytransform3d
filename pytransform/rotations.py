import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.testing import assert_array_almost_equal


unitx = np.array([1.0, 0.0, 0.0])
unity = np.array([0.0, 1.0, 0.0])
unitz = np.array([0.0, 0.0, 1.0])


def matrix_from_angle_axis(a):
    ua = a / np.linalg.norm(a)
    theta, ux, uy, uz = ua
    cost = np.cos(theta)
    costi = 1.0 - cost
    sint = np.sin(theta)

    R = np.array([[ux * ux * costi + cost,
                   ux * uy * costi - uz * sint,
                   ux * uz * costi + uy * sint],
                  [uy * ux * costi + uz * sint,
                   uy * uy * costi + cost,
                   uy * uz * costi - ux * sint],
                  [uz * ux * costi - uy * sint,
                   uz * uy * costi + ux * sint,
                   uz * uz * costi + cost],
                  ])
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


def quaternion_from_angle_axis(a):
    ua = a / np.linalg.norm(a)
    theta = ua[0]
    q = np.empty(4)
    q[0] = np.cos(theta / 2)
    q[1:] = np.sin(theta / 2) * ua[1:]
    return q


def check_rotation_matrix(R):
    assert_array_almost_equal(np.dot(R, R.T), np.eye(3))
    assert_array_almost_equal(np.linalg.det(R), 1.0)


def plot_basis(ax, R, p=None, s=1.0, **kwargs):
    if ax is None:
        plt.subplot(111, projection="3d", aspect="equal")
    if p is None:
        p = np.zeros(3)

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
