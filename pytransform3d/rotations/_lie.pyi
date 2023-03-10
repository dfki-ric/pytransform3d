import numpy as np
import numpy.typing as npt


def left_jacobian_SO3(omega: npt.ArrayLike) -> np.ndarray: ...


def left_jacobian_SO3_inv(omega: npt.ArrayLike) -> np.ndarray: ...
