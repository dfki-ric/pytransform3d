import numpy as np
import numpy.typing as npt


def wedge(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def plane_normal_from_bivector(B: npt.ArrayLike) -> np.ndarray: ...


def geometric_product(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray: ...


def rotor_reverse(rotor: npt.ArrayLike) -> np.ndarray: ...


def concatenate_rotors(
        rotor1: npt.ArrayLike, rotor2: npt.ArrayLike) -> np.ndarray: ...


def rotor_apply(rotor: npt.ArrayLike, v: npt.ArrayLike) -> np.ndarray: ...


def matrix_from_rotor(rotor: npt.ArrayLike) -> np.ndarray: ...


def rotor_from_two_directions(
        v_from: npt.ArrayLike, v_to: npt.ArrayLike) -> np.ndarray: ...


def rotor_from_plane_angle(B: npt.ArrayLike, angle: float) -> np.ndarray: ...
