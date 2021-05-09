import numpy as np
import numpy.typing as npt


def assert_axis_angle_equal(a1: npt.ArrayLike, a2: npt.ArrayLike, *args, **kwargs): ...


def assert_compact_axis_angle_equal(a1: npt.ArrayLike, a2: npt.ArrayLike, *args, **kwargs): ...


def assert_quaternion_equal(q1: npt.ArrayLike, q2: npt.ArrayLike, *args, **kwargs): ...


def assert_euler_xyz_equal(e_xyz1: npt.ArrayLike, e_xyz2: npt.ArrayLike, *args, **kwargs): ...


def assert_euler_zyx_equal(e_zyx1: npt.ArrayLike, e_zyx2: npt.ArrayLike, *args, **kwargs): ...


def assert_rotation_matrix(R: npt.ArrayLike, *args, **kwargs): ...
