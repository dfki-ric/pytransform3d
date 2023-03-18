import numpy as np
import numpy.typing as npt


def jacobian_SE3(Stheta: npt.ArrayLike) -> np.ndarray: ...


def jacobian_SE3_series(Stheta: npt.ArrayLike, n_terms: int) -> np.ndarray: ...


def jacobian_SE3_inv(Stheta: npt.ArrayLike) -> np.ndarray: ...


def jacobian_SE3_inv_series(Stheta: npt.ArrayLike, n_terms: int) -> np.ndarray: ...
