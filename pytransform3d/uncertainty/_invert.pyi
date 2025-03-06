from typing import Tuple

import numpy as np
import numpy.typing as npt

def invert_uncertain_transform(
    mean: npt.ArrayLike, cov: npt.ArrayLike
) -> Tuple[np.ndarray, np.ndarray]: ...
