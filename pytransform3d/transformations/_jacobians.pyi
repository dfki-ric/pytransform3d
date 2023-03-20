import numpy as np
import numpy.typing as npt


def left_jacobian_SE3(Stheta: npt.ArrayLike) -> np.ndarray: ...


def left_jacobian_SE3_series(Stheta: npt.ArrayLike, n_terms: int) -> np.ndarray: ...


def left_jacobian_SE3_inv(Stheta: npt.ArrayLike) -> np.ndarray: ...


def left_jacobian_SE3_inv_series(Stheta: npt.ArrayLike, n_terms: int) -> np.ndarray: ...
