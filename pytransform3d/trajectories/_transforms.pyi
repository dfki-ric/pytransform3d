import numpy as np
import numpy.typing as npt


def invert_transforms(A2Bs: npt.ArrayLike) -> np.ndarray: ...


def concat_one_to_many(
        A2B: npt.ArrayLike, B2Cs: npt.ArrayLike) -> np.ndarray: ...


def concat_many_to_one(
        A2Bs: npt.ArrayLike, B2C: npt.ArrayLike) -> np.ndarray: ...

def concat_many_to_many(
        A2B: npt.ArrayLike, B2C: npt.ArrayLike) -> np.ndarray: ...

def concat_dynamic(
        A2B: npt.ArrayLike, B2C: npt.ArrayLike) -> np.ndarray: ...
