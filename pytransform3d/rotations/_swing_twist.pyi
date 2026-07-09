from typing import Tuple

import numpy as np
import numpy.typing as npt

def swing_twist_decomposition(
    q: npt.ArrayLike, axis: npt.ArrayLike, eps: float = ...
) -> Tuple[np.ndarray, np.ndarray]: ...
def swing_twist_composition(
    swing: npt.ArrayLike, twist: npt.ArrayLike
) -> np.ndarray: ...
