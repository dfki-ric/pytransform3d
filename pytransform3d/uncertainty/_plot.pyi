import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from mpl_toolkits.mplot3d import Axes3D

def to_ellipsoid(
    mean: npt.ArrayLike, cov: npt.ArrayLike
) -> Tuple[np.ndarray, np.ndarray]: ...
def to_projected_ellipsoid(
    mean: npt.ArrayLike,
    cov: npt.ArrayLike,
    factor: float = ...,
    n_steps: int = ...,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
def plot_projected_ellipsoid(
    ax: Union[None, Axes3D],
    mean: npt.ArrayLike,
    cov: npt.ArrayLike,
    factor: float = ...,
    wireframe: bool = ...,
    n_steps: int = ...,
    color: str = ...,
    alpha: float = ...,
) -> Axes3D: ...
