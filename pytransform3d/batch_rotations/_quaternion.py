import numpy as np

from ._axis_angle import norm_axis_angles
from ._utils import norm_vectors
from ..rotations import (
    angle_between_vectors,
    slerp_weights,
    pick_closest_quaternion,
)


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
        np.logical_and(
            before_smooth_jump_indices[:-1], before_smooth_jump_indices[1:]
        )
    )[0]
    before_jump_indices = np.unique(
        np.hstack((before_jump_indices, before_smooth_jump_indices))
    ).tolist()
    before_jump_indices.append(len(Q) - 1)

    slices_to_correct = np.array(
        list(zip(before_jump_indices[:-1], before_jump_indices[1:]))
    )[::2]
    for i, j in slices_to_correct:
        Q[i + 1 : j + 1] *= -1.0
    return Q


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
        raise ValueError(
            "Number of dimensions must be the same. "
            "Got %d for Q1 and %d for Q2." % (Q1.ndim, Q2.ndim)
        )
    for d in range(Q1.ndim - 1):
        if Q1.shape[d] != Q2.shape[d]:
            raise ValueError(
                "Size of dimension %d does not match: %d != %d"
                % (d + 1, Q1.shape[d], Q2.shape[d])
            )
    if Q1.shape[-1] != 4:
        raise ValueError(
            "Last dimension of first argument does not match. A quaternion "
            "must have 4 entries, got %d" % Q1.shape[-1]
        )
    if Q2.shape[-1] != 4:
        raise ValueError(
            "Last dimension of second argument does not match. A quaternion "
            "must have 4 entries, got %d" % Q2.shape[-1]
        )

    if out is None:
        out = np.empty_like(Q1)

    vector_inner_products = np.sum(Q1[..., 1:] * Q2[..., 1:], axis=-1)
    out[..., 0] = Q1[..., 0] * Q2[..., 0] - vector_inner_products
    out[..., 1:] = (
        Q1[..., 0, np.newaxis] * Q2[..., 1:]
        + Q2[..., 0, np.newaxis] * Q1[..., 1:]
        + np.cross(Q1[..., 1:], Q2[..., 1:])
    )
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
    return (
        w1[:, np.newaxis] * start[np.newaxis]
        + w2[:, np.newaxis] * end[np.newaxis]
    )


def axis_angles_from_quaternions(qs):
    """Compute axis-angle from quaternion.

    This operation is called logarithmic map.

    We usually assume active rotations.

    Parameters
    ----------
    qs : array-like, shape (..., 4)
        Unit quaternion to represent rotation: (w, x, y, z)

    Returns
    -------
    as : array, shape (..., 4)
        Axis of rotation and rotation angle: (x, y, z, angle). The angle is
        constrained to [0, pi) so that the mapping is unique.
    """
    quaternion_vector_part = qs[..., 1:]
    qvec_norm = np.linalg.norm(quaternion_vector_part, axis=-1)

    # Vectorized branches bases on norm of the vector part
    small_p_norm_mask = qvec_norm < np.finfo(float).eps
    non_zero_mask = ~small_p_norm_mask

    axes = (
        quaternion_vector_part[non_zero_mask]
        / qvec_norm[non_zero_mask, np.newaxis]
    )

    w_clamped = np.clip(qs[non_zero_mask, 0], -1.0, 1.0)
    angles = 2.0 * np.arccos(w_clamped)

    result = np.empty_like(qs)
    result[non_zero_mask] = norm_axis_angles(
        np.concatenate((axes, angles[..., np.newaxis]), axis=-1)
    )
    result[small_p_norm_mask] = np.array([1.0, 0.0, 0.0, 0.0])

    return result


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
