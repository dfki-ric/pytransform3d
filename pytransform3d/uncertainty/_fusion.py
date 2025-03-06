"""Fusion of poses."""

import numpy as np

from ..transformations import (
    exponential_coordinates_from_transform,
    invert_transform,
    left_jacobian_SE3_inv,
    transform_from_exponential_coordinates,
)


def pose_fusion(means, covs):
    """Fuse Gaussian distributions of multiple poses.

    Parameters
    ----------
    means : array-like, shape (n_poses, 4, 4)
        Homogeneous transformation matrices.

    covs : array-like, shape (n_poses, 6, 6)
        Covariances of pose distributions in exponential coordinate space.

    Returns
    -------
    mean : array, shape (4, 4)
        Fused pose mean.

    cov : array, shape (6, 6)
        Fused pose covariance.

    V : float
        Error of optimization objective.

    References
    ----------
    .. [1] Barfoot, T. D., Furgale, P. T. (2014).
       Associating Uncertainty With Three-Dimensional Poses for Use in
       Estimation Problems. IEEE Transactions on Robotics, 30(3), pp. 679-693,
       doi: 10.1109/TRO.2014.2298059.
    """
    n_poses = len(means)
    covs_inv = [np.linalg.inv(cov) for cov in covs]

    mean = np.eye(4)
    LHS = np.empty((6, 6))
    RHS = np.empty(6)
    for _ in range(20):
        LHS[:, :] = 0.0
        RHS[:] = 0.0
        for k in range(n_poses):
            x_ik = exponential_coordinates_from_transform(
                np.dot(mean, invert_transform(means[k]))
            )
            J_inv = left_jacobian_SE3_inv(x_ik)
            J_invT_S = np.dot(J_inv.T, covs_inv[k])
            LHS += np.dot(J_invT_S, J_inv)
            RHS += np.dot(J_invT_S, x_ik)
        x_i = np.linalg.solve(-LHS, RHS)
        mean = np.dot(transform_from_exponential_coordinates(x_i), mean)

    cov = np.linalg.inv(LHS)

    V = 0.0
    for k in range(n_poses):
        x_ik = exponential_coordinates_from_transform(
            np.dot(mean, invert_transform(means[k]))
        )
        V += 0.5 * np.dot(x_ik, np.dot(covs_inv[k], x_ik))
    return mean, cov, V
