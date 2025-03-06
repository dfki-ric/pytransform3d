import numpy as np


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

    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))

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
    angle_zero = angles == 0.0
    angle_not_zero = np.logical_not(angle_zero)

    Rs_diag = np.einsum("nii->ni", Rs.reshape(-1, 3, 3))
    if instances_shape:
        Rs_diag = Rs_diag.reshape(*(instances_shape + (3,)))
    else:
        Rs_diag = Rs_diag[0]

    out[angle_close_to_pi, :3] = np.sqrt(
        0.5 * (Rs_diag[angle_close_to_pi] + 1.0)
    ) * np.sign(out[angle_close_to_pi, :3])
    out[angle_not_zero, :3] /= np.linalg.norm(out[angle_not_zero, :3], axis=-1)[
        ..., np.newaxis
    ]

    out[angle_zero, 0] = 1.0
    out[angle_zero, 1:3] = 0.0

    out[..., 3] = angles

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
        np.logical_and(
            Rs[..., 0, 0] > Rs[..., 1, 1], Rs[..., 0, 0] > Rs[..., 2, 2]
        ),
    )
    s = 2.0 * np.sqrt(1.0 + Rs[ind2, 0, 0] - Rs[ind2, 1, 1] - Rs[ind2, 2, 2])
    out[ind2, 0] = (Rs[ind2, 2, 1] - Rs[ind2, 1, 2]) / s
    out[ind2, 1] = 0.25 * s
    out[ind2, 2] = (Rs[ind2, 1, 0] + Rs[ind2, 0, 1]) / s
    out[ind2, 3] = (Rs[ind2, 0, 2] + Rs[ind2, 2, 0]) / s

    ind3 = np.logical_and(np.logical_not(ind1), Rs[..., 1, 1] > Rs[..., 2, 2])
    s = 2.0 * np.sqrt(1.0 + Rs[ind3, 1, 1] - Rs[ind3, 0, 0] - Rs[ind3, 2, 2])
    out[ind3, 0] = (Rs[ind3, 0, 2] - Rs[ind3, 2, 0]) / s
    out[ind3, 1] = (Rs[ind3, 1, 0] + Rs[ind3, 0, 1]) / s
    out[ind3, 2] = 0.25 * s
    out[ind3, 3] = (Rs[ind3, 2, 1] + Rs[ind3, 1, 2]) / s

    ind4 = np.logical_and(
        np.logical_and(np.logical_not(ind1), np.logical_not(ind2)),
        np.logical_not(ind3),
    )
    s = 2.0 * np.sqrt(1.0 + Rs[ind4, 2, 2] - Rs[ind4, 0, 0] - Rs[ind4, 1, 1])
    out[ind4, 0] = (Rs[ind4, 1, 0] - Rs[ind4, 0, 1]) / s
    out[ind4, 1] = (Rs[ind4, 0, 2] + Rs[ind4, 2, 0]) / s
    out[ind4, 2] = (Rs[ind4, 2, 1] + Rs[ind4, 1, 2]) / s
    out[ind4, 3] = 0.25 * s

    return out
