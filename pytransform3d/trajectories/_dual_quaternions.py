import numpy as np

from ..batch_rotations import (
    batch_concatenate_quaternions,
    batch_q_conj,
    axis_angles_from_quaternions,
    matrices_from_quaternions,
)
from ._screws import (
    dual_quaternions_from_screw_parameters,
)


def batch_dq_conj(dqs):
    """Conjugate of dual quaternions.

    There are three different conjugates for dual quaternions. The one that we
    use here converts (pw, px, py, pz, qw, qx, qy, qz) to
    (pw, -px, -py, -pz, -qw, qx, qy, qz). It is a combination of the quaternion
    conjugate and the dual number conjugate.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_conjugates : array-like, shape (..., 8)
        Conjugates of dual quaternions: (pw, -px, -py, -pz, -qw, qx, qy, qz)
    """
    out = np.empty_like(dqs)
    out[..., 0] = dqs[..., 0]
    out[..., 1:5] = -dqs[..., 1:5]
    out[..., 5:] = dqs[..., 5:]
    return out


def batch_dq_q_conj(dqs):
    """Quaternion conjugate of dual quaternions.

    For unit dual quaternions that represent transformations,
    this function is equivalent to the inverse of the
    corresponding transformation matrix.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dq_q_conjugates : array, shape (..., 8)
        Conjugate of dual quaternion: (pw, -px, -py, -pz, qw, -qx, -qy, -qz)

    See Also
    --------
    pytransform3d.transformations.dq_q_conj
        Quaternion conjugate of dual quaternions.
    """
    out = np.empty_like(dqs)
    out[..., 0] = dqs[..., 0]
    out[..., 1:4] = -dqs[..., 1:4]
    out[..., 4] = dqs[..., 4]
    out[..., 5:8] = -dqs[..., 5:8]
    return out


def batch_concatenate_dual_quaternions(dqs1, dqs2):
    """Concatenate dual quaternions.

    Suppose we want to apply two extrinsic transforms given by dual
    quaternions dq1 and dq2 to a vector v. We can either apply dq2 to v and
    then dq1 to the result or we can concatenate dq1 and dq2 and apply the
    result to v.

    Parameters
    ----------
    dqs1 : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    dqs2 : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    dqs3 : array, shape (8,)
        Products of the two batches of dual quaternions:
        (pw, px, py, pz, qw, qx, qy, qz)
    """
    dqs1 = np.asarray(dqs1)
    dqs2 = np.asarray(dqs2)

    out = np.empty_like(dqs1)
    out[..., :4] = batch_concatenate_quaternions(dqs1[..., :4], dqs2[..., :4])
    out[..., 4:] = batch_concatenate_quaternions(
        dqs1[..., :4], dqs2[..., 4:]
    ) + batch_concatenate_quaternions(dqs1[..., 4:], dqs2[..., :4])
    return out


def batch_dq_prod_vector(dqs, V):
    """Apply transforms represented by a dual quaternions to vectors.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternions

    V : array-like, shape (..., 3)
        3d vectors

    Returns
    -------
    W : array, shape (3,)
        3d vectors
    """
    dqs = np.asarray(dqs)

    v_dqs = np.empty_like(dqs)
    v_dqs[..., 0] = 1.0
    v_dqs[..., 1:5] = 0.0
    v_dqs[..., 5:] = V
    v_dq_transformed = batch_concatenate_dual_quaternions(
        batch_concatenate_dual_quaternions(dqs, v_dqs), batch_dq_conj(dqs)
    )
    return v_dq_transformed[..., 5:]


def dual_quaternions_power(dqs, ts):
    """Compute power of unit dual quaternions with respect to scalar.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternions to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    t : array-like, shape (...)
        Exponent

    Returns
    -------
    dq_ts : array, shape (..., 8)
        Unit dual quaternions to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz) ** t

    See Also
    --------
    pytransform3d.transformations.dual_quaternion_power :
        Compute power of unit dual quaternion with respect to scalar.
    """
    q, s_axis, h, theta = screw_parameters_from_dual_quaternions(dqs)
    return dual_quaternions_from_screw_parameters(q, s_axis, h, theta * ts)


def dual_quaternions_sclerp(starts, ends, ts):
    """Screw linear interpolation (ScLERP) for array of dual quaternions.

    Parameters
    ----------
    starts : array-like, shape (..., 8)
        Unit dual quaternion to represent start poses:
        (pw, px, py, pz, qw, qx, qy, qz)

    end : array-like, shape (..., 8)
        Unit dual quaternion to represent end poses:
        (pw, px, py, pz, qw, qx, qy, qz)

    ts : array-like, shape (...)
        Positions between starts and goals

    Returns
    -------
    dq_ts : array, shape (..., 8)
        Interpolated unit dual quaternion: (pw, px, py, pz, qw, qx, qy, qz)


    See Also
    --------
    pytransform3d.transformations.dual_quaternion_sclerp :
        Screw linear interpolation (ScLERP) for dual quaternions.
    """
    starts = np.asarray(starts)
    ends = np.asarray(ends)
    ts = np.asarray(ts)

    if starts.shape != ends.shape:
        raise ValueError(
            "The 'starts' and 'ends' arrays must have the same shape."
        )

    if ts.ndim != starts.ndim - 1 or (
        ts.ndim > 0 and ts.shape != starts.shape[:-1]
    ):
        raise ValueError(
            "ts array, shape=%s must have the same number of elements as "
            "starts array, shape=%s" % (ts.shape, starts.shape)
        )

    diffs = batch_concatenate_dual_quaternions(batch_dq_q_conj(starts), ends)
    powers = dual_quaternions_power(diffs, ts)
    return batch_concatenate_dual_quaternions(starts, powers)


def pqs_from_dual_quaternions(dqs):
    """Get positions and quaternions from dual quaternions.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    pqs : array, shape (..., 7)
        Poses represented by positions and quaternions in the
        order (x, y, z, qw, qx, qy, qz)
    """
    dqs = np.asarray(dqs)
    instances_shape = dqs.shape[:-1]
    out = np.empty(instances_shape + (7,))
    out[..., 3:] = dqs[..., :4]
    out[..., :3] = (
        2
        * batch_concatenate_quaternions(
            dqs[..., 4:], batch_q_conj(out[..., 3:])
        )[..., 1:]
    )
    return out


def screw_parameters_from_dual_quaternions(dqs):
    """Compute screw parameters from dual quaternions.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Unit dual quaternion to represent transform:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    qs : array, shape (..., 3)
        Vector to a point on the screw axis

    s_axiss : array, shape (..., 3)
        Direction vector of the screw axis

    hs : array, shape (...,)
        Pitch of the screw. The pitch is the ratio of translation and rotation
        of the screw axis. Infinite pitch indicates pure translation.

    thetas : array, shape (...,)
        Parameter of the transformation: theta is the angle of rotation
        and h * theta the translation.

    See Also
    --------
    pytransform3d.transformations.screw_parameters_from_dual_quaternion :
        Compute screw parameters from dual quaternion.
    """
    reals = dqs[..., :4]
    duals = dqs[..., 4:]

    a = axis_angles_from_quaternions(reals)
    s_axis = np.copy(a[..., :3])
    thetas = a[..., 3]

    translation = (
        2 * batch_concatenate_quaternions(duals, batch_q_conj(reals))[..., 1:]
    )

    # instead of the if/else stamenets in the
    # screw_parameters_from_dual_quaternion function
    # we use mask array to enable vectorized operations
    # the name of the mask represent the according block in
    # the original function
    outer_if_mask = np.abs(thetas) < np.finfo(float).eps
    outer_else_mask = np.logical_not(outer_if_mask)

    ds = np.linalg.norm(translation, axis=-1)
    inner_if_mask = ds < np.finfo(float).eps

    outer_if_inner_if_mask = np.logical_and(outer_if_mask, inner_if_mask)
    outer_if_inner_else_mask = np.logical_and(
        outer_if_mask, np.logical_not(inner_if_mask)
    )

    if np.any(outer_if_inner_if_mask):
        s_axis[outer_if_inner_if_mask] = np.array([1.0, 0.0, 0.0])

    if np.any(outer_if_inner_else_mask):
        s_axis[outer_if_inner_else_mask] = (
            translation[outer_if_inner_else_mask]
            / ds[outer_if_inner_else_mask][..., np.newaxis]
        )

    qs = np.zeros(dqs.shape[:-1] + (3,))
    thetas[outer_if_mask] = ds[outer_if_mask]
    hs = np.full(dqs.shape[:-1], np.inf)

    if np.any(outer_else_mask):
        distance = np.einsum(
            "ij,ij->i", translation[outer_else_mask], s_axis[outer_else_mask]
        )

        moment = 0.5 * (
            np.cross(translation[outer_else_mask], s_axis[outer_else_mask])
            + (
                translation[outer_else_mask]
                - distance[..., np.newaxis] * s_axis[outer_else_mask]
            )
            / np.tan(0.5 * thetas[outer_else_mask])[..., np.newaxis]
        )

        qs[outer_else_mask] = np.cross(s_axis[outer_else_mask], moment)
        hs[outer_else_mask] = distance / thetas[outer_else_mask]

    return qs, s_axis, hs, thetas


def transforms_from_dual_quaternions(dqs):
    """Get transformations from dual quaternions.

    Parameters
    ----------
    dqs : array-like, shape (..., 8)
        Dual quaternions to represent transforms:
        (pw, px, py, pz, qw, qx, qy, qz)

    Returns
    -------
    A2Bs : array, shape (..., 4, 4)
        Poses represented by homogeneous matrices
    """
    dqs = np.asarray(dqs)
    instances_shape = dqs.shape[:-1]
    out = np.empty(instances_shape + (4, 4))
    out[..., :3, :3] = matrices_from_quaternions(dqs[..., :4])
    out[..., :3, 3] = (
        2
        * batch_concatenate_quaternions(
            dqs[..., 4:], batch_q_conj(dqs[..., :4])
        )[..., 1:]
    )
    out[..., 3, :3] = 0.0
    out[..., 3, 3] = 1.0
    return out
