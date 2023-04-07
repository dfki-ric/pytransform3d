import numpy as np
import numpy.typing as npt
from typing import Tuple, Union
from mpl_toolkits.mplot3d import Axes3D


def estimate_gaussian_transform_from_samples(
        samples: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def invert_uncertain_transform(
        mean: npt.ArrayLike,
        cov: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def concat_globally_uncertain_transforms(
        mean_A2B: npt.ArrayLike, cov_A2B: npt.ArrayLike,
        mean_B2C: npt.ArrayLike, cov_B2C: npt.ArrayLike) -> np.ndarray: ...


def concat_locally_uncertain_transforms(
        mean_A2B: npt.ArrayLike, mean_B2C: npt.ArrayLike, cov_A: npt.ArrayLike,
        cov_B: npt.ArrayLike) -> np.ndarray: ...


def pose_fusion(means: npt.ArrayLike, covs: npt.ArrayLike) -> np.ndarray: ...


def to_ellipsoid(mean: npt.ArrayLike,
                 cov: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def to_projected_ellipsoid(
        mean: npt.ArrayLike, cov: npt.ArrayLike,
        factor: float = ..., n_steps: int = ...
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def plot_projected_ellipsoid(
    ax: Union[None, Axes3D],
    mean: npt.ArrayLike, cov: npt.ArrayLike,
    factor: float = ..., wireframe: bool = ..., n_steps: int = ...,
    color: str = ..., alpha: float = ...
) -> Axes3D: ...
