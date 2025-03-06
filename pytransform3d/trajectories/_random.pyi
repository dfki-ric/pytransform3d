import numpy as np
import numpy.typing as npt

def random_trajectories(
    rng: np.random.Generator = ...,
    n_trajectories: int = ...,
    n_steps: int = ...,
    start: npt.ArrayLike = ...,
    goal: npt.ArrayLike = ...,
    scale: npt.ArrayLike = ...,
) -> np.ndarray: ...
def _linear_movement(
    start: npt.ArrayLike, goal: npt.ArrayLike, n_steps: int, dt: float
) -> np.ndarray: ...
def _acceleration_L(n_dims: int, n_steps: int, dt: float) -> np.ndarray: ...
def _create_fd_matrix_1d(n_steps: int, dt: float) -> np.ndarray: ...
