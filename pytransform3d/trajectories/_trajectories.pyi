import numpy as np
import numpy.typing as npt
from mpl_toolkits.mplot3d import Axes3D
from typing import Union

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


def screw_parameters_from_dual_quaternions(
        dqs: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternions_from_screw_parameters(
        qs: npt.ArrayLike,
        s_axis: npt.ArrayLike,
        hs: npt.ArrayLike,
        thetas: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternions_power(
        dqs: npt.ArrayLike,
        ts: npt.ArrayLike) -> np.ndarray: ...


def dual_quaternions_sclerp(
        starts: npt.ArrayLike,
        ends: npt.ArrayLike,
        ts: npt.ArrayLike) -> np.ndarray: ...


def transforms_from_dual_quaternions(dqs: npt.ArrayLike) -> np.ndarray: ...


def batch_dq_conj(dqs: npt.ArrayLike) -> np.ndarray: ...


def batch_dq_q_conj(dqs: npt.ArrayLike) -> np.ndarray: ...


def batch_concatenate_dual_quaternions(
        dqs1: npt.ArrayLike, dqs2: npt.ArrayLike) -> np.ndarray: ...


def batch_dq_prod_vector(
        dqs: npt.ArrayLike, V: npt.ArrayLike) -> np.ndarray: ...


def random_trajectories(
        rng: np.random.Generator = ..., n_trajectories: int = ...,
        n_steps: int = ..., start: npt.ArrayLike = ...,
        goal: npt.ArrayLike = ..., scale: npt.ArrayLike = ...
) -> np.ndarray: ...


def _linear_movement(
        start: npt.ArrayLike, goal:npt.ArrayLike, n_steps: int, dt: float
) -> np.ndarray: ...


def _acceleration_L(n_dims: int, n_steps: int, dt: float) -> np.ndarray: ...


def _create_fd_matrix_1d(n_steps: int, dt: float) -> np.ndarray: ...


def plot_trajectory(
        ax: Union[None, Axes3D] = ..., P: Union[None, npt.ArrayLike] = ...,
        normalize_quaternions: bool = ..., show_direction: bool = ...,
        n_frames: int = ..., s: float = ..., ax_s: float = ...,
        **kwargs) -> Axes3D: ...


def mirror_screw_axis_direction(Sthetas: npt.ArrayLike) -> np.ndarray: ...
