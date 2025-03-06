import numpy as np
import numpy.typing as npt

def concat_globally_uncertain_transforms(
    mean_A2B: npt.ArrayLike,
    cov_A2B: npt.ArrayLike,
    mean_B2C: npt.ArrayLike,
    cov_B2C: npt.ArrayLike,
) -> np.ndarray: ...
def concat_locally_uncertain_transforms(
    mean_A2B: npt.ArrayLike,
    mean_B2C: npt.ArrayLike,
    cov_A: npt.ArrayLike,
    cov_B: npt.ArrayLike,
) -> np.ndarray: ...
