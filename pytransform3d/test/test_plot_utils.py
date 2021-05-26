import numpy as np
try:
    import matplotlib
except ImportError:
    from nose import SkipTest
    raise SkipTest("matplotlib is required for these tests")
from pytransform3d.plot_utils import (
    make_3d_axis, remove_frame, Frame, LabeledFrame, Trajectory)
from nose.tools import assert_equal, assert_less


def test_make_3d_axis():
    ax = make_3d_axis(2.0)
    try:
        assert_equal(ax.name, "3d")
        assert_equal(ax.get_xlim(), (-2, 2))
        assert_equal(ax.get_ylim(), (-2, 2))
        assert_equal(ax.get_zlim(), (-2, 2))
    finally:
        ax.remove()


def test_make_3d_axis_subplots():
    ax1 = make_3d_axis(1.0, 121)
    ax2 = make_3d_axis(1.0, 122)
    try:
        bounds1 = ax1.get_position().bounds
        bounds2 = ax2.get_position().bounds
        assert_less(bounds1[0], bounds2[0])
        assert_less(bounds1[2], bounds2[2])
    finally:
        ax1.remove()
        ax2.remove()


def test_make_3d_axis_with_units():
    ax = make_3d_axis(1.0, unit="m")
    try:
        assert_equal(ax.get_xlabel(), "X [m]")
        assert_equal(ax.get_ylabel(), "Y [m]")
        assert_equal(ax.get_zlabel(), "Z [m]")
    finally:
        ax.remove()


def test_frame_removed():
    ax = make_3d_axis(1.0)
    try:
        remove_frame(ax)
        bounds = ax.get_position().bounds
        # regression test
        assert_equal(bounds, (0.125, 0.0, 0.75, 1.0))
    finally:
        ax.remove()


def test_frame():
    ax = make_3d_axis(1)
    try:
        frame = Frame(np.eye(4), label="Frame", s=0.1)
        frame.add_frame(ax)
        assert_equal(len(ax.lines), 4)  # 3 axes and black line to text
        assert_equal(len(ax.texts), 1)
    finally:
        ax.remove()


def test_labeled_frame():
    ax = make_3d_axis(1)
    try:
        frame = LabeledFrame(np.eye(4), label="Frame", s=0.1)
        frame.add_frame(ax)
        assert_equal(len(ax.lines), 4)  # 3 axes and black line to text
        assert_equal(len(ax.texts), 4)
    finally:
        ax.remove()


def test_trajectory():
    ax = make_3d_axis(1)
    try:
        trajectory = Trajectory(
            np.array([np.eye(4), np.eye(4)]), s=0.1, n_frames=2)
        trajectory.add_trajectory(ax)
        assert_equal(len(ax.lines), 7)  # 2 * 3 axes + connection line
        assert_equal(len(ax.artists), 1)  # arrow
    finally:
        ax.remove()
