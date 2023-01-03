import numpy as np
import numpy.typing as npt


def quaternion_integrate(Qd: npt.ArrayLike, q0: npt.ArrayLike = ..., dt: float = ...) -> np.ndarray: ...


def quaternion_gradient(Q: npt.ArrayLike, dt: float = ...) -> np.ndarray: ...


def concatenate_quaternions(q1: npt.ArrayLike, q2: npt.ArrayLike) -> np.ndarray: ...


def q_prod_vector(q: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray: ...


def q_conj(q: npt.ArrayLike) -> np.ndarray: ...


def quaternion_dist(q1: npt.ArrayLike, q2: npt.ArrayLike) -> np.ndarray: ...


def quaternion_diff(q1: npt.ArrayLike, q2: npt.ArrayLike) -> np.ndarray: ...


def quaternion_from_euler(
        e: npt.ArrayLike, i: int, j: int, k: int, extrinsic: bool) -> np.ndarray: ...
