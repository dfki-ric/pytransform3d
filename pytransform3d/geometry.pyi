import numpy as np
import numpy.typing as npt
from typing import Tuple


class GeometricShape(object):
    pose: np.ndarray

    def __init__(self, pose: npt.ArrayLike): ...


class Sphere(GeometricShape):
    def __init__(self, pose: npt.ArrayLike, radius: npt.ArrayLike): ...

    def surface(self, n_steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
