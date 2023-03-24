import numpy as np
import numpy.typing as npt
from typing import Tuple


def estimate_gaussian_transform_from_samples(
        samples: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def invert_uncertain_transform(
        mean: npt.ArrayLike,
        cov: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def concat_uncertain_transforms(
        mean_A2B: npt.ArrayLike, cov_A2B: npt.ArrayLike,
        mean_B2C: npt.ArrayLike, cov_B2C: npt.ArrayLike) -> np.ndarray: ...


def pose_fusion(means: npt.ArrayLike, covs: npt.ArrayLike) -> np.ndarray: ...


def to_ellipsoid(mean: npt.ArrayLike,
                 cov: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def to_projected_ellipsoid(
        mean: npt.ArrayLike, cov: npt.ArrayLike,
        factor: float = ..., n_steps: int = ...
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
