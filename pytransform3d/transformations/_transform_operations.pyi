import numpy as np
import numpy.typing as npt


def invert_transform(A2B: npt.ArrayLike, strict_check: bool = ...,
                     check: bool = ...) -> np.ndarray: ...


def vector_to_point(v: npt.ArrayLike) -> np.ndarray: ...


def vectors_to_points(V: npt.ArrayLike) -> np.ndarray: ...


def vector_to_direction(v: npt.ArrayLike) -> np.ndarray: ...


def vectors_to_directions(V: npt.ArrayLike) -> np.ndarray: ...


def concat(A2B: npt.ArrayLike, B2C: npt.ArrayLike, strict_check: bool = ...,
           check: bool = ...) -> np.ndarray: ...


def transform(A2B: npt.ArrayLike, PA: npt.ArrayLike,
              strict_check: bool = ...) -> np.ndarray: ...


def scale_transform(
        A2B: npt.ArrayLike, s_xr: float = ..., s_yr: float = ...,
        s_zr: float = ..., s_r: float = ..., s_xt: float = ...,
        s_yt: float = ..., s_zt: float = ..., s_t: float = ...,
        s_d: float = ..., strict_check: bool = ...) -> np.ndarray: ...
