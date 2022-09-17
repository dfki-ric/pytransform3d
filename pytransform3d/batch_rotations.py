"""Batch operations on rotations in three dimensions - SO(3).

Conversions from this module operate on batches of orientations or rotations
and can be orders of magnitude faster than a loop of individual conversions.

All functions operate on nd arrays, where the last dimension (vectors) or
the last two dimensions (matrices) contain individual rotations.
"""
import numpy as np
from .rotations import (
    angle_between_vectors, slerp_weights, pick_closest_quaternion)


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


def angles_between_vectors(A, B):
    """Compute angle between two vectors.

    Parameters
    ----------
    A : array-like, shape (..., n)
        nd vectors

    B : array-like, shape (..., n)
        nd vectors

    Returns
    -------
    angles : array, shape (...)
        Angles between pairs of vectors from A and B
    """
    A = np.asarray(A)
    B = np.asarray(B)
    n_dims = A.shape[-1]
    A_norms = np.linalg.norm(A, axis=-1)
    B_norms = np.linalg.norm(B, axis=-1)
    AdotB = np.einsum(
        "ni,ni->n", A.reshape(-1, n_dims), B.reshape(-1, n_dims)
    ).reshape(A.shape[:-1])
    return np.arccos(np.clip(AdotB / (A_norms * B_norms), -1.0, 1.0))


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


def active_matrices_from_intrinsic_euler_angles(
        basis1, basis2, basis3, e, out=None):
    """Compute active rotation matrices from intrinsic Euler angles.

    Parameters
    ----------
    basis1 : int
        Basis vector of first rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis2 : int
        Basis vector of second rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis3 : int
        Basis vector of third rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    e : array-like, shape (..., 3)
        Euler angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    e = np.asarray(e)
    R_shape = e.shape + (3,)
    R_alpha = active_matrices_from_angles(basis1, e[..., 0].flat)
    R_beta = active_matrices_from_angles(basis2, e[..., 1].flat)
    R_gamma = active_matrices_from_angles(basis3, e[..., 2].flat)

    if out is None:
        out = np.empty(R_shape)

    out[:] = np.einsum(
        "nij,njk->nik", np.einsum("nij,njk->nik", R_alpha, R_beta),
        R_gamma).reshape(R_shape)

    return out


def active_matrices_from_extrinsic_euler_angles(
        basis1, basis2, basis3, e, out=None):
    """Compute active rotation matrices from extrinsic Euler angles.

    Parameters
    ----------
    basis1 : int
        Basis vector of first rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis2 : int
        Basis vector of second rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    basis3 : int
        Basis vector of third rotation. 0 corresponds to x axis, 1 to y axis,
        and 2 to z axis.

    e : array-like, shape (..., 3)
        Euler angles

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    e = np.asarray(e)
    R_shape = e.shape + (3,)
    R_alpha = active_matrices_from_angles(basis1, e[..., 0].flat)
    R_beta = active_matrices_from_angles(basis2, e[..., 1].flat)
    R_gamma = active_matrices_from_angles(basis3, e[..., 2].flat)

    if out is None:
        out = np.empty(R_shape)

    out[:] = np.einsum(
        "nij,njk->nik", np.einsum("nij,njk->nik", R_gamma, R_beta),
        R_alpha).reshape(R_shape)

    return out


def matrices_from_compact_axis_angles(
        A=None, axes=None, angles=None, out=None):
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

    out[..., 0] = Rs[..., 2, 1] - Rs[..., 1, 2]
    out[..., 1] = Rs[..., 0, 2] - Rs[..., 2, 0]
    out[..., 2] = Rs[..., 1, 0] - Rs[..., 0, 1]

    # The threshold is a result from this discussion:
    # https://github.com/dfki-ric/pytransform3d/issues/43
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
    out[angle_close_to_pi, :3] = (
        np.sqrt(0.5 * (Rs_diag[angle_close_to_pi] + 1.0))
        * np.sign(out[angle_close_to_pi, :3]))
    out[angle_not_zero, :3] /= np.linalg.norm(
        out[angle_not_zero, :3], axis=-1)[..., np.newaxis]

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


def matrices_from_quaternions(Q, normalize_quaternions=True, out=None):
    """Compute rotation matrices from quaternions.

    Parameters
    ----------
    Q : array-like, shape (..., 4)
        Unit quaternions to represent rotations: (w, x, y, z)

    normalize_quaternions : bool, optional (default: True)
        Normalize quaternions before conversion

    out : array, shape (..., 3, 3), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Rs : array, shape (..., 3, 3)
        Rotation matrices
    """
    Q = np.asarray(Q)

    if normalize_quaternions:
        Q = norm_vectors(Q)

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
    """Compute quaternions from rotation matrices.

    Parameters
    ----------
    Rs : array-like, shape (..., 3, 3)
        Rotation matrices

    out : array, shape (..., 4), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Q : array, shape (..., 4)
        Unit quaternions to represent rotations: (w, x, y, z)
    """
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


def quaternion_slerp_batch(start, end, t, shortest_path=False):
    """Spherical linear interpolation for a batch of steps.

    Parameters
    ----------
    start : array-like, shape (4,)
        Start unit quaternion to represent rotation: (w, x, y, z)

    end : array-like, shape (4,)
        End unit quaternion to represent rotation: (w, x, y, z)

    t : array-like, shape (n_steps,)
        Steps between start and goal, must be in interval [0, 1]

    shortest_path : bool, optional (default: False)
        Resolve sign ambiguity before interpolation to find the shortest path.
        The end quaternion will be picked to be close to the start quaternion.

    Returns
    -------
    Q : array, shape (n_steps, 4)
        Interpolated unit quaternions
    """
    t = np.asarray(t)
    if shortest_path:
        end = pick_closest_quaternion(end, start)
    angle = angle_between_vectors(start, end)
    w1, w2 = slerp_weights(angle, t)
    return (w1[:, np.newaxis] * start[np.newaxis]
            + w2[:, np.newaxis] * end[np.newaxis])


def batch_concatenate_quaternions(Q1, Q2, out=None):
    """Concatenate two batches of quaternions.

    We use Hamilton's quaternion multiplication.

    Suppose we want to apply two extrinsic rotations given by quaternions
    q1 and q2 to a vector v. We can either apply q2 to v and then q1 to
    the result or we can concatenate q1 and q2 and apply the result to v.

    Parameters
    ----------
    Q1 : array-like, shape (..., 4)
        First batch of quaternions

    Q2 : array-like, shape (..., 4)
        Second batch of quaternions

    out : array, shape (..., 4), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Q12 : array, shape (..., 4)
        Batch of quaternions that represents the concatenated rotations

    Raises
    ------
    ValueError
        If the input dimensions are incorrect
    """
    Q1 = np.asarray(Q1)
    Q2 = np.asarray(Q2)

    if Q1.ndim != Q2.ndim:
        raise ValueError("Number of dimensions must be the same. "
                         "Got %d for Q1 and %d for Q2." % (Q1.ndim, Q2.ndim))
    for d in range(Q1.ndim - 1):
        if Q1.shape[d] != Q2.shape[d]:
            raise ValueError(
                "Size of dimension %d does not match: %d != %d"
                % (d + 1, Q1.shape[d], Q2.shape[d]))
    if Q1.shape[-1] != 4:
        raise ValueError(
            "Last dimension of first argument does not match. A quaternion "
            "must have 4 entries, got %d" % Q1.shape[-1])
    if Q2.shape[-1] != 4:
        raise ValueError(
            "Last dimension of second argument does not match. A quaternion "
            "must have 4 entries, got %d" % Q2.shape[-1])

    if out is None:
        out = np.empty_like(Q1)

    vector_inner_products = np.sum(Q1[..., 1:] * Q2[..., 1:], axis=-1)
    out[..., 0] = Q1[..., 0] * Q2[..., 0] - vector_inner_products
    out[..., 1:] = (Q1[..., 0, np.newaxis] * Q2[..., 1:] +
                    Q2[..., 0, np.newaxis] * Q1[..., 1:] +
                    np.cross(Q1[..., 1:], Q2[..., 1:]))
    return out


def batch_q_conj(Q):
    """Conjugate of quaternions.

    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion. The conjugate of a quaternion q is often denoted
    as q*.

    Parameters
    ----------
    Q : array-like, shape (..., 4)
        Unit quaternions to represent rotations: (w, x, y, z)

    Returns
    -------
    Q_c : array, shape (..., 4,)
        Conjugates (w, -x, -y, -z)
    """
    Q = np.asarray(Q)
    out = np.empty_like(Q)
    out[..., 0] = Q[..., 0]
    out[..., 1:] = -Q[..., 1:]
    return out


def batch_quaternion_wxyz_from_xyzw(Q_xyzw, out=None):
    """Converts from x, y, z, w to w, x, y, z convention.

    Parameters
    ----------
    Q_xyzw : array-like, shape (..., 4)
        Quaternions with scalar part after vector part

    out : array, shape (..., 4), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Q_wxyz : array-like, shape (..., 4)
        Quaternions with scalar part before vector part
    """
    Q_xyzw = np.asarray(Q_xyzw)
    if out is None:
        out = np.empty_like(Q_xyzw)
    out[..., 0] = Q_xyzw[..., 3]
    out[..., 1] = Q_xyzw[..., 0]
    out[..., 2] = Q_xyzw[..., 1]
    out[..., 3] = Q_xyzw[..., 2]
    return out


def batch_quaternion_xyzw_from_wxyz(Q_wxyz, out=None):
    """Converts from w, x, y, z to x, y, z, w convention.

    Parameters
    ----------
    Q_wxyz : array-like, shape (..., 4)
        Quaternions with scalar part before vector part

    out : array, shape (..., 4), optional (default: new array)
        Output array to which we write the result

    Returns
    -------
    Q_xyzw : array-like, shape (..., 4)
        Quaternions with scalar part after vector part
    """
    Q_wxyz = np.asarray(Q_wxyz)
    if out is None:
        out = np.empty_like(Q_wxyz)
    out[..., 0] = Q_wxyz[..., 1]
    out[..., 1] = Q_wxyz[..., 2]
    out[..., 2] = Q_wxyz[..., 3]
    out[..., 3] = Q_wxyz[..., 0]
    return out


def smooth_quaternion_trajectory(Q, start_component_positive="x"):
    """Smooth quaternion trajectory.

    Quaternion q and -q represent the same rotation but cannot be
    interpolated well. This function guarantees that two successive
    quaternions q1 and q2 are closer than q1 and -q2.

    Parameters
    ----------
    Q : array-like, shape (n_steps, 4)
        Unit quaternions to represent rotations: (w, x, y, z)

    start_component_positive : str, optional (default: 'x')
        Start trajectory with quaternion that has this component positive.
        Allowed values: 'w' (scalar), 'x', 'y', and 'z'.

    Returns
    -------
    Q : array, shape (n_steps, 4)
        Unit quaternions to represent rotations: (w, x, y, z)

    Raises
    ------
    ValueError
        If Q has length 0.
    """
    Q = np.copy(Q)

    if len(Q) == 0:
        raise ValueError("At least one quaternion is expected.")

    if Q[0, "wxyz".index(start_component_positive)] < 0.0:
        Q[0] *= -1.0

    q1q2_dists = np.linalg.norm(Q[:-1] - Q[1:], axis=1)
    q1mq2_dists = np.linalg.norm(Q[:-1] + Q[1:], axis=1)
    before_jump_indices = np.where(q1q2_dists > q1mq2_dists)[0]

    # workaround for interpolation artifacts:
    before_smooth_jump_indices = np.isclose(q1q2_dists, q1mq2_dists)
    before_smooth_jump_indices = np.where(
        np.logical_and(before_smooth_jump_indices[:-1],
                       before_smooth_jump_indices[1:]))[0]
    before_jump_indices = np.unique(
        np.hstack((before_jump_indices, before_smooth_jump_indices))).tolist()
    before_jump_indices.append(len(Q) - 1)

    slices_to_correct = np.array(
        list(zip(before_jump_indices[:-1], before_jump_indices[1:])))[::2]
    for i, j in slices_to_correct:
        Q[i + 1:j + 1] *= -1.0
    return Q
