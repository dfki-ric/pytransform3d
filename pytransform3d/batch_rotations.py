import numpy as np
from .rotations import angle_between_vectors


def norm_vectors(V, out=None):
    """Normalize vectors.

    Parameters
    ----------
    V : array-like, shape (..., n)
        nd vectors

    out : array, shape (..., n), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    V_unit : array, shape (..., n)
        nd unit vectors with norm 1 or zero vectors
    """
    V = np.asarray(V)
    norms = np.linalg.norm(V, axis=-1)
    if out is None:
        out = np.empty_like(V)
    # Avoid division by zero with np.maximum(..., smallest positive float).
    # The norm is zero only when the vector is zero so this case does not
    # require further processing.
    out[...] = V / np.maximum(norms[..., np.newaxis], np.finfo(float).tiny)
    return out


def active_matrices_from_angles(basis, angles, out=None):
    """Compute active rotation matrices from rotation about basis vectors.

    Parameters
    ----------
    basis : int from [0, 1, 2]
        The rotation axis (0: x, 1: y, 2: z)

    angles : array-like, shape (...)
        Rotation angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    angles = np.asarray(angles)
    c = np.cos(angles)
    s = np.sin(angles)

    R_shape = angles.shape + (3, 3)
    if out is None:
        out = np.empty(R_shape)

    out[..., basis, :] = 0.0
    out[..., :, basis] = 0.0
    out[..., basis, basis] = 1.0
    basisp1 = (basis + 1) % 3
    basisp2 = (basis + 2) % 3
    out[..., basisp1, basisp1] = c
    out[..., basisp2, basisp2] = c
    out[..., basisp1, basisp2] = -s
    out[..., basisp2, basisp1] = s

    return out


def active_matrices_from_intrinsic_euler_angles(basis1, basis2, basis3, e, out=None):
    e = np.asarray(e)
    R_shape = e.shape + (3,)
    R_alpha = active_matrices_from_angles(basis1, e[..., 0].flat)
    R_beta = active_matrices_from_angles(basis2, e[..., 1].flat)
    R_gamma = active_matrices_from_angles(basis3, e[..., 2].flat)
    R = np.einsum("nij,njk->nik", np.einsum("nij,njk->nik", R_alpha, R_beta), R_gamma)

    if out is None:
        out = np.empty(R_shape)

    out[:] = R.reshape(R_shape)
    return out


def active_matrices_from_extrinsic_euler_angles(basis1, basis2, basis3, e, out=None):
    e = np.asarray(e)
    R_shape = e.shape + (3,)
    R_alpha = active_matrices_from_angles(basis1, e[..., 0].flat)
    R_beta = active_matrices_from_angles(basis2, e[..., 1].flat)
    R_gamma = active_matrices_from_angles(basis3, e[..., 2].flat)
    R = np.einsum("nij,njk->nik", np.einsum("nij,njk->nik", R_gamma, R_beta), R_alpha)

    if out is None:
        out = np.empty(R_shape)

    out[:] = R.reshape(R_shape)
    return out


def matrices_from_compact_axis_angles(A=None, axes=None, angles=None, out=None):
    """Compute rotation matrices from compact axis-angle representations.

    This is called exponential map or Rodrigues' formula.

    This typically results in an active rotation matrix.

    Parameters
    ----------
    A : array-like, shape (..., 3)
        Axes of rotation and rotation angles in compact representation:
        angle * (x, y, z)

    axes : array, shape (..., 3)
        If the unit axes of rotation have been precomputed, you can pass them
        here.

    angles : array, shape (...)
        If the angles have been precomputed, you can pass them here.

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    # TODO test
    if angles is None:
        thetas = np.linalg.norm(A, axis=-1)
    else:
        thetas = np.asarray(angles)

    if axes is None:
        omega_unit = norm_vectors(A)
    else:
        omega_unit = axes

    c = np.cos(thetas)
    s = np.sin(thetas)
    ci = 1.0 - c
    ux = omega_unit[..., 0]
    uy = omega_unit[..., 1]
    uz = omega_unit[..., 2]

    uxs = ux * s
    uys = uy * s
    uzs = uz * s
    ciux = ci * ux
    ciuy = ci * uy
    ciuxuy = ciux * uy
    ciuxuz = ciux * uz
    ciuyuz = ciuy * uz

    if out is None:
        out = np.empty(A.shape[:-1] + (3, 3))

    out[..., 0, 0] = ciux * ux + c
    out[..., 0, 1] = ciuxuy - uzs
    out[..., 0, 2] = ciuxuz + uys
    out[..., 1, 0] = ciuxuy + uzs
    out[..., 1, 1] = ciuy * uy + c
    out[..., 1, 2] = ciuyuz - uxs
    out[..., 2, 0] = ciuxuz - uys
    out[..., 2, 1] = ciuyuz + uxs
    out[..., 2, 2] = ci * uz * uz + c

    return out


def axis_angles_from_matrices(Rs, traces=None, out=None):
    """Compute compact axis-angle representations from rotation matrices.

    This is called logarithmic map.

    Parameters
    ----------
    Rs : array-like, shape (..., 3, 3)
        Rotation matrices

    traces : array, shape (..., 3)
        If the traces of rotation matrices been precomputed, you can pass them
        here.

    out : array, shape (..., 4), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    A : array, shape (..., 4)
        Axes of rotation and rotation angles: (x, y, z, angle)
    """
    Rs = np.asarray(Rs)

    instances_shape = Rs.shape[:-2]

    if traces is None:
        traces = np.einsum("nii", Rs.reshape(-1, 3, 3))
        if instances_shape:
            traces = traces.reshape(*instances_shape)
        else:
            # this works because indX will be a single boolean and
            # out[True, n] = value will assign value to out[n], while
            # out[False, n] = value will not assign value to out[n]
            traces = traces[0]

    angles = np.arccos((traces - 1.0) / 2.0)

    if out is None:
        out = np.empty(instances_shape + (4,))

    # We can usually determine the rotation axis by inverting Rodrigues'
    # formula. Subtracting opposing off-diagonal elements gives us
    # 2 * sin(angle) * e,
    # where e is the normalized rotation axis.
    out[..., 0] = Rs[..., 2, 1] - Rs[..., 1, 2]
    out[..., 1] = Rs[..., 0, 2] - Rs[..., 2, 0]
    out[..., 2] = Rs[..., 1, 0] - Rs[..., 0, 1]

    # The threshold is a result from this discussion:
    # https://github.com/rock-learning/pytransform3d/issues/43
    # The standard formula becomes numerically unstable, however,
    # Rodrigues' formula reduces to R = I + 2 (ee^T - I), with the
    # rotation axis e, that is, ee^T = 0.5 * (R + I) and we can find the
    # squared values of the rotation axis on the diagonal of this matrix.
    # We can still use the original formula to reconstruct the signs of
    # the rotation axis correctly.
    angle_close_to_pi = np.abs(angles - np.pi) < 1e-4
    angle_not_zero = np.abs(angles) != 0.0

    Rs_diag = np.einsum("nii->ni", Rs.reshape(-1, 3, 3))
    if instances_shape:
        Rs_diag = Rs_diag.reshape(*(instances_shape + (3,)))
    else:
        Rs_diag = Rs_diag[0]
    out[angle_close_to_pi, :3] = np.sqrt(0.5 * (Rs_diag[angle_close_to_pi] + 1.0)) * np.sign(out[angle_close_to_pi, :3])
    # The norm of axis_unnormalized is 2.0 * np.sin(angle), that is, we
    # could normalize with a[:3] = a[:3] / (2.0 * np.sin(angle)),
    # but the following is much more precise for angles close to 0 or pi:
    out[angle_not_zero, :3] /= np.linalg.norm(out[angle_not_zero, :3], axis=-1)[..., np.newaxis]

    out[..., 3] = angles

    return out


def cross_product_matrices(V):
    """Generate the cross-product matrices of vectors.

    The cross-product matrix :math:`\\boldsymbol{V}` satisfies the equation

    .. math::

        \\boldsymbol{V} \\boldsymbol{w} = \\boldsymbol{v} \\times
        \\boldsymbol{w}

    It is a skew-symmetric (antisymmetric) matrix, i.e.
    :math:`-\\boldsymbol{V} = \\boldsymbol{V}^T`.

    Parameters
    ----------
    V : array-like, shape (..., 3)
        3d vectors

    Returns
    -------
    V_cross_product_matrices : array, shape (..., 3, 3)
        Cross-product matrices of V
    """
    V = np.asarray(V)

    instances_shape = V.shape[:-1]
    V_matrices = np.empty(instances_shape + (3, 3))

    V_matrices[..., 0, 0] = 0.0
    V_matrices[..., 0, 1] = -V[..., 2]
    V_matrices[..., 0, 2] = V[..., 1]
    V_matrices[..., 1, 0] = V[..., 2]
    V_matrices[..., 1, 1] = 0.0
    V_matrices[..., 1, 2] = -V[..., 0]
    V_matrices[..., 2, 0] = -V[..., 1]
    V_matrices[..., 2, 1] = V[..., 0]
    V_matrices[..., 2, 2] = 0.0

    return V_matrices


def matrices_from_quaternions(Q, out=None):  # only normalized quaternions!
    Q = np.asarray(Q)

    w = Q[..., 0]
    x = Q[..., 1]
    y = Q[..., 2]
    z = Q[..., 3]

    x2 = 2.0 * x * x
    y2 = 2.0 * y * y
    z2 = 2.0 * z * z
    xy = 2.0 * x * y
    xz = 2.0 * x * z
    yz = 2.0 * y * z
    xw = 2.0 * x * w
    yw = 2.0 * y * w
    zw = 2.0 * z * w

    if out is None:
        out = np.empty(w.shape + (3, 3))

    out[..., 0, 0] = 1.0 - y2 - z2
    out[..., 0, 1] = xy - zw
    out[..., 0, 2] = xz + yw
    out[..., 1, 0] = xy + zw
    out[..., 1, 1] = 1.0 - x2 - z2
    out[..., 1, 2] = yz - xw
    out[..., 2, 0] = xz - yw
    out[..., 2, 1] = yz + xw
    out[..., 2, 2] = 1.0 - x2 - y2

    return out


def quaternions_from_matrices(Rs, out=None):
    Rs = np.asarray(Rs)
    instances_shape = Rs.shape[:-2]

    if out is None:
        out = np.empty(instances_shape + (4,))

    traces = np.einsum("nii", Rs.reshape(-1, 3, 3))
    if instances_shape:
        traces = traces.reshape(*instances_shape)
    else:
        # this works because indX will be a single boolean and
        # out[True, n] = value will assign value to out[n], while
        # out[False, n] = value will not assign value to out[n]
        traces = traces[0]
    ind1 = traces > 0.0
    s = 2.0 * np.sqrt(1.0 + traces[ind1])
    out[ind1, 0] = 0.25 * s
    out[ind1, 1] = (Rs[ind1, 2, 1] - Rs[ind1, 1, 2]) / s
    out[ind1, 2] = (Rs[ind1, 0, 2] - Rs[ind1, 2, 0]) / s
    out[ind1, 3] = (Rs[ind1, 1, 0] - Rs[ind1, 0, 1]) / s

    ind2 = np.logical_and(
        np.logical_not(ind1),
        np.logical_and(Rs[..., 0, 0] > Rs[..., 1, 1],
                       Rs[..., 0, 0] > Rs[..., 2, 2]))
    s = 2.0 * np.sqrt(1.0 + Rs[ind2, 0, 0] - Rs[ind2, 1, 1] - Rs[ind2, 2, 2])
    out[ind2, 0] = (Rs[ind2, 2, 1] - Rs[ind2, 1, 2]) / s
    out[ind2, 1] = 0.25 * s
    out[ind2, 2] = (Rs[ind2, 1, 0] + Rs[ind2, 0, 1]) / s
    out[ind2, 3] = (Rs[ind2, 0, 2] + Rs[ind2, 2, 0]) / s

    ind3 = np.logical_and(
        np.logical_not(ind1), Rs[..., 1, 1] > Rs[..., 2, 2])
    s = 2.0 * np.sqrt(1.0 + Rs[ind3, 1, 1] - Rs[ind3, 0, 0] - Rs[ind3, 2, 2])
    out[ind3, 0] = (Rs[ind3, 0, 2] - Rs[ind3, 2, 0]) / s
    out[ind3, 1] = (Rs[ind3, 1, 0] + Rs[ind3, 0, 1]) / s
    out[ind3, 2] = 0.25 * s
    out[ind3, 3] = (Rs[ind3, 2, 1] + Rs[ind3, 1, 2]) / s

    ind4 = np.logical_and(
        np.logical_and(np.logical_not(ind1),
                       np.logical_not(ind2)),
        np.logical_not(ind3))
    s = 2.0 * np.sqrt(1.0 + Rs[ind4, 2, 2] - Rs[ind4, 0, 0] - Rs[ind4, 1, 1])
    out[ind4, 0] = (Rs[ind4, 1, 0] - Rs[ind4, 0, 1]) / s
    out[ind4, 1] = (Rs[ind4, 0, 2] + Rs[ind4, 2, 0]) / s
    out[ind4, 2] = (Rs[ind4, 2, 1] + Rs[ind4, 1, 2]) / s
    out[ind4, 3] = 0.25 * s

    return out


def quaternion_slerp_batch(start, end, t):
    """Spherical linear interpolation for a batch of steps.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start unit quaternion to represent rotation: (w, x, y, z)

    end : array-like, shape (4,)
        End unit quaternion to represent rotation: (w, x, y, z)

    t : array-like, shape (n_steps,)
        Steps between start and goal, must be in interval [0, 1]

    Returns
    -------
    q : array-like, shape (n_steps, 4)
        Interpolated unit quaternions
    """
    angle = angle_between_vectors(start, end)
    w1, w2 = _slerp_weights(angle, t)
    return w1[:, np.newaxis] * start[np.newaxis] + w2[:, np.newaxis] * end[np.newaxis]


def _slerp_weights(angle, t):
    if angle == 0.0:
        return np.ones_like(t), np.zeros_like(t)
    else:
        return (np.sin((1.0 - t) * angle) / np.sin(angle),
                np.sin(t * angle) / np.sin(angle))


def angles_between_vectors(A, B):  # TODO test einsum
    """Compute angle between two vectors.

    Parameters
    ----------
    A : array-like, shape (n_vectors, n)
        nd vector

    B : array-like, shape (n_vectors, n)
        nd vector

    fast : bool, optional (default: False)
        Use fast implementation instead of numerically stable solution

    Returns
    -------
    angles : array, shape (n_vectors,)
        Angles between pairs of vectors from A and B
    """
    A_norms = np.linalg.norm(A, axis=1)
    B_norms = np.linalg.norm(B, axis=1)
    return np.arccos(np.clip(np.einsum("ni,ni->ni", A, B) / (A_norms * B_norms)), -1.0, 1.0)
