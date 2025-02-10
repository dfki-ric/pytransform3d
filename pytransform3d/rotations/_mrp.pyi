import numpy as np
import numpy.typing as npt


def norm_mrp(mrp: npt.ArrayLike) -> np.ndarray: ...


def check_mrp(mrp: npt.ArrayLike) -> np.ndarray: ...


def mrp_near_singularity(mrp: npt.ArrayLike, tolerance: float = ...) -> bool: ...


def mrp_double(mrp: npt.ArrayLike) -> np.ndarray: ...


def assert_mrp_equal(mrp1: npt.ArrayLike, mrp2: npt.ArrayLike, *args, **kwargs): ...


def concatenate_mrp(mrp1: npt.ArrayLike, mrp2: npt.ArrayLike) -> np.ndarray: ...


def mrp_prod_vector(mrp: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_mrp(mrp: npt.ArrayLike) -> np.ndarray: ...


def axis_angle_from_mrp(mrp: npt.ArrayLike) -> np.ndarray: ...
