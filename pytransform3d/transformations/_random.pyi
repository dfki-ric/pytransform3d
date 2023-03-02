import numpy as np
import numpy.typing as npt


def random_transform(
        rng: np.random.Generator = ...,
        mean: npt.ArrayLike = ...,
        cov: npt.ArrayLike = ...
) -> np.ndarray: ...


def random_screw_axis(rng: np.random.Generator = ...) -> np.ndarray: ...


def random_exponential_coordinates(
        rng: np.random.Generator = ...,
        cov: npt.ArrayLike = ...
) -> np.ndarray: ...
