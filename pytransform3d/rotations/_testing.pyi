import numpy as np
import numpy.typing as npt


def assert_euler_equal(e1: npt.ArrayLike, e2: npt.ArrayLike, i: int, j: int, k: int, *args, **kwargs): ...


def assert_axis_angle_equal(a1: npt.ArrayLike, a2: npt.ArrayLike, *args, **kwargs): ...


def assert_compact_axis_angle_equal(a1: npt.ArrayLike, a2: npt.ArrayLike, *args, **kwargs): ...


def assert_quaternion_equal(q1: npt.ArrayLike, q2: npt.ArrayLike, *args, **kwargs): ...


def assert_rotation_matrix(R: npt.ArrayLike, *args, **kwargs): ...
