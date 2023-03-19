import numpy as np
import numpy.typing as npt
from typing import Tuple


def invert_uncertain_transform(mean: npt.ArrayLike, cov: npt.ArrayLike) -> Tuple[np.ndarray, np.ndarray]: ...


def pose_fusion(means: npt.ArrayLike, covs: npt.ArrayLike) -> np.ndarray: ...
