from typing import Tuple
import numpy as np
import numpy.typing as npt


def unit_sphere_surface_grid(
        n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def transform_surface(
        pose: npt.ArrayLike, x: npt.ArrayLike, y: npt.ArrayLike,
        z: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
