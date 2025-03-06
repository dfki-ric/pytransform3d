"""Random trajectory generation."""

import numpy as np

from ._dual_quaternions import (
    transforms_from_dual_quaternions,
    dual_quaternions_sclerp,
)
from ._screws import (
    transforms_from_exponential_coordinates,
)
from ._transforms import (
    concat_many_to_many,
    dual_quaternions_from_transforms,
)

_N_EXP_COORDINATE_DIMS = 6


def random_trajectories(
    rng=np.random.default_rng(0),
    n_trajectories=10,
    n_steps=101,
    start=np.eye(4),
    goal=np.eye(4),
    scale=100.0 * np.ones(6),
):
    """Generate random trajectories.

    Create a smooth random trajectory with low accelerations.

    The generated trajectories consist of a linear movement from start to goal
    and a superimposed random movement with low accelerations. Hence, the first
    pose and last pose do not exactly equal the start and goal pose
    respectively.

    Parameters
    ----------
    rng : np.random.Generator, optional (default: random seed 0)
        Random number generator

    n_trajectories : int, optional (default: 10)
        Number of trajectories that should be generated.

    n_steps : int, optional (default: 101)
        Number of steps in each trajectory.

    start : array-like, shape (4, 4), optional (default: I)
        Start pose as transformation matrix.

    goal : array-like, shape (4, 4), optional (default: I)
        Goal pose as transformation matrix.

    scale : array-like, shape (6,), optional (default: [100] * 6)
        Scaling factor for random deviations from linear movement from start to
        goal.

    Returns
    -------
    trajectories : array, shape (n_trajectories, n_steps, 4, 4)
        Random trajectories between start and goal.
    """
    dt = 1.0 / (n_steps - 1)
    linear_component = _linear_movement(start, goal, n_steps, dt)

    L = _acceleration_L(_N_EXP_COORDINATE_DIMS, n_steps, dt)
    samples = rng.normal(
        size=(n_trajectories, _N_EXP_COORDINATE_DIMS * n_steps)
    )
    smooth_samples = np.dot(samples, L.T)
    Sthetas = smooth_samples.reshape(
        n_trajectories, _N_EXP_COORDINATE_DIMS, n_steps
    ).transpose([0, 2, 1])
    Sthetas *= np.asarray(scale)[np.newaxis, np.newaxis]

    trajectories = transforms_from_exponential_coordinates(Sthetas)
    for i in range(n_trajectories):
        trajectories[i] = concat_many_to_many(trajectories[i], linear_component)

    return trajectories


def _linear_movement(start, goal, n_steps, dt):
    """Linear movement from start to goal.

    Parameters
    ----------
    start : array-like, shape (4, 4)
        Start pose as transformation matrix.

    goal : array-like, shape (4, 4)
        Goal pose as transformation matrix.

    n_steps : int
        Number of steps.

    dt : float
        Time difference between two steps.

    Returns
    -------
    linear_component : array, shape (n_steps, 4, 4)
        Linear trajectory from start to goal with equal step sizes.
    """
    time = np.arange(n_steps) * dt
    start_dq = dual_quaternions_from_transforms(start)
    goal_dq = dual_quaternions_from_transforms(goal)
    return transforms_from_dual_quaternions(
        dual_quaternions_sclerp(
            np.repeat(start_dq[np.newaxis], n_steps, axis=0),
            np.repeat(goal_dq[np.newaxis], n_steps, axis=0),
            time,
        )
    )


def _acceleration_L(n_dims, n_steps, dt):
    """Cholesky decomposition of a smooth trajectory covariance.

    Parameters
    ----------
    n_dims : int
        Number of dimensions.

    n_steps : int
        Number of steps in the trajectory.

    dt : float
        Time difference between two steps.

    Returns
    -------
    L : array, shape (n_steps * n_dims, n_steps * n_dims)
        Cholesky decomposition of covariance created from finite difference
        matrix.
    """
    A_per_dim = _create_fd_matrix_1d(n_steps, dt)
    covariance = np.linalg.inv(np.dot(A_per_dim.T, A_per_dim))
    L_per_dim = np.linalg.cholesky(covariance)

    # Copy L for each dimension
    L = np.zeros((n_dims * n_steps, n_dims * n_steps))
    for d in range(n_dims):
        L[d * n_steps : (d + 1) * n_steps, d * n_steps : (d + 1) * n_steps] = (
            L_per_dim
        )
    return L


def _create_fd_matrix_1d(n_steps, dt):
    r"""Create one-dimensional finite difference matrix for second derivative.

    The finite difference matrix A for the second derivative with respect to
    time is defined as:

    .. math::

        \ddot(x) = A x

    Parameters
    ----------
    n_steps : int
        Number of steps in the trajectory

    dt : float
        Time in seconds between successive steps

    Returns
    -------
    A : array, shape (n_steps + 2, n_steps)
        Finite difference matrix for second derivative with respect to time
    """
    A = np.zeros((n_steps + 2, n_steps), dtype=float)
    super_diagonal = (np.arange(n_steps), np.arange(n_steps))
    A[super_diagonal] = np.ones(n_steps)
    sub_diagonal = (np.arange(2, n_steps + 2), np.arange(n_steps))
    A[sub_diagonal] = np.ones(n_steps)
    main_diagonal = (np.arange(1, n_steps + 1), np.arange(n_steps))
    A[main_diagonal] = -2 * np.ones(n_steps)
    return A / (dt**2)
