import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union


def invert_transforms(A2Bs: npt.ArrayLike) -> np.ndarray: ...


def concat_one_to_many(
    A2B: npt.ArrayLike, B2Cs: npt.ArrayLike) -> np.ndarray: ...


def concat_many_to_one(
    A2Bs: npt.ArrayLike, B2C: npt.ArrayLike) -> np.ndarray: ...


def transforms_from_pqs(
        P: npt.ArrayLike, normalize_quaternions: bool = ...) -> np.ndarray: ...


def pqs_from_transforms(A2Bs: npt.ArrayLike) -> np.ndarray: ...


def exponential_coordinates_from_transforms(
        A2Bs: npt.ArrayLike) -> np.ndarray: ...


def transforms_from_exponential_coordinates(
        Sthetas: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternions_from_pqs(pqs: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternions_from_transforms(A2Bs: npt.ArrayLike) -> np.ndarray: ...


def pqs_from_dual_quaternions(dqs: npt.ArrayLike) -> np.ndarray: ...


def transforms_from_dual_quaternions(dqs: npt.ArrayLike) -> np.ndarray: ...


def batch_dq_conj(dqs: npt.ArrayLike) -> np.ndarray: ...


def batch_concatenate_dual_quaternions(
        dqs1: npt.ArrayLike, dqs2: npt.ArrayLike) -> np.ndarray: ...


def batch_dq_prod_vector(dqs: npt.ArrayLike, V: npt.ArrayLike) -> np.ndarray: ...


def plot_trajectory(
        ax: Union[None, Axes3D] = ..., P: Union[None, npt.ArrayLike] = ...,
        normalize_quaternions: bool = ..., show_direction: bool = ...,
        n_frames: int = ..., s: float = ..., ax_s: float = ...,
        **kwargs) -> Axes3D: ...


def mirror_screw_axis_direction(Sthetas: npt.ArrayLike) -> np.ndarray: ...
