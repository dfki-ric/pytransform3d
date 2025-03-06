import numpy as np
import numpy.typing as npt

def mirror_screw_axis_direction(Sthetas: npt.ArrayLike) -> np.ndarray: ...
def transforms_from_exponential_coordinates(
    Sthetas: npt.ArrayLike,
) -> np.ndarray: ...
def dual_quaternions_from_screw_parameters(
    qs: npt.ArrayLike,
    s_axis: npt.ArrayLike,
    hs: npt.ArrayLike,
    thetas: npt.ArrayLike,
) -> np.ndarray: ...
