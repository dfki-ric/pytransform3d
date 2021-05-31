import numpy as np
import numpy.typing as npt


def axis_angle_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def quaternion_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def rotor_slerp(
        start: npt.ArrayLike, end: npt.ArrayLike, t: float) -> np.ndarray: ...


def slerp_weights(angle: float, t: float) -> np.ndarray: ...
