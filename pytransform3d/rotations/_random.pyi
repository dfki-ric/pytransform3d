import numpy as np
import numpy.typing as npt


def random_vector(rng: np.random.Generator = ..., n: int = ...) -> np.ndarray: ...


def random_axis_angle(rng: np.random.Generator = ...) -> np.ndarray: ...


def random_compact_axis_angle(rng: np.random.Generator = ...) -> np.ndarray: ...


def random_quaternion(rng: np.random.Generator = ...) -> np.ndarray: ...
