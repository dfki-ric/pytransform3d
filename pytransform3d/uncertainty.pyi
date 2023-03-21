import numpy as np
import numpy.typing as npt
from typing import Tuple


def invert_uncertain_transform(mean: npt.ArrayLike, cov: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def pose_fusion(means: npt.ArrayLike, covs: npt.ArrayLike) -> np.ndarray: ...


def concat_uncertain_transforms(
        mean_A2B: npt.ArrayLike, cov_A2B: npt.ArrayLike,
        mean_B2C: npt.ArrayLike, cov_B2C: npt.ArrayLike) -> np.ndarray: ...


def estimate_gaussian_transform_from_samples(
        samples: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...
