import numpy as np
import numpy.typing as npt


def left_jacobian_SO3(omega: npt.ArrayLike) -> np.ndarray: ...


def left_jacobian_SO3_series(omega: npt.ArrayLike, n_terms: int) -> np.ndarray: ...


def left_jacobian_SO3_inv(omega: npt.ArrayLike) -> np.ndarray: ...


def left_jacobian_SO3_inv_series(omega: npt.ArrayLike, n_terms: int) -> np.ndarray: ...
