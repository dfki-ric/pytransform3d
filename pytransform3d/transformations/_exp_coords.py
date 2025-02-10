"""Exponential coordinates of transformation."""
import numpy as np
from ._conversions import (
    screw_parameters_from_screw_axis, screw_axis_from_screw_parameters)
from ..rotations import norm_angle, eps


def norm_exponential_coordinates(Stheta):
    """Normalize exponential coordinates of transformation.

    Parameters
    ----------
    Stheta : array-like, shape (6,)
        Exponential coordinates of transformation:
        S * theta = (omega_1, omega_2, omega_3, v_1, v_2, v_3) * theta,
        where the first 3 components are related to rotation and the last 3
        components are related to translation. Theta is the rotation angle
        and h * theta the translation. Theta should be >= 0. Negative rotations
        will be represented by a negative screw axis instead. This is relevant
        if you want to recover theta from exponential coordinates.

    Returns
    -------
    Stheta : array, shape (6,)
        Normalized exponential coordinates of transformation with theta in
        [0, pi]. Note that in the case of pure translation no normalization
        is required because the representation is unique. In the case of
        rotation by pi, there is an ambiguity that will be resolved so that
        the screw pitch is positive.
    """
    theta = np.linalg.norm(Stheta[:3])
    if theta == 0.0:
        return Stheta

    screw_axis = Stheta / theta
    q, s_axis, h = screw_parameters_from_screw_axis(screw_axis)
    if abs(theta - np.pi) < eps and h < 0:
        h *= -1.0
        s_axis *= -1.0
    theta_normed = norm_angle(theta)
    h_normalized = h * theta / theta_normed
    screw_axis = screw_axis_from_screw_parameters(q, s_axis, h_normalized)

    return screw_axis * theta_normed
