import numpy as np
import numpy.typing as npt


def assert_transform(A2B: npt.ArrayLike, *args, **kwargs): ...


def assert_unit_dual_quaternion(dq: npt.ArrayLike, *args, **kwargs): ...


def assert_unit_dual_quaternion_equal(dq1: npt.ArrayLike, dq2: npt.ArrayLike, *args, **kwargs): ...


def assert_screw_parameters_equal(
        q1: npt.ArrayLike, s_axis1: npt.ArrayLike, h1: float, theta1: float,
        q2: npt.ArrayLike, s_axis2: npt.ArrayLike, h2: float, theta2: float,
        *args, **kwargs): ...
