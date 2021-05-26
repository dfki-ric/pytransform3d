import numpy as np
from nose import SkipTest
try:
    import matplotlib
except ImportError:
    raise SkipTest("matplotlib is required for these tests")
from pytransform3d.plot_utils import (
    make_3d_axis, remove_frame, Frame, LabeledFrame, Trajectory,
    plot_box, plot_sphere, plot_cylinder, plot_mesh, plot_vector,
    plot_length_variable)
from nose.tools import assert_equal, assert_less, assert_greater_equal


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
    ax = make_3d_axis(1.0)
    try:
        frame = Frame(np.eye(4), label="Frame", s=0.1)
        frame.add_frame(ax)
        assert_equal(len(ax.lines), 4)  # 3 axes and black line to text
        assert_equal(len(ax.texts), 1)  # label
    finally:
        ax.remove()


def test_labeled_frame():
    ax = make_3d_axis(1.0)
    try:
        frame = LabeledFrame(np.eye(4), label="Frame", s=0.1)
        frame.add_frame(ax)
        assert_equal(len(ax.lines), 4)  # 3 axes and black line to text
        assert_equal(len(ax.texts), 4)  # label and 3 axis labels
    finally:
        ax.remove()


def test_trajectory():
    ax = make_3d_axis(1.0)
    try:
        trajectory = Trajectory(
            np.array([np.eye(4), np.eye(4)]), s=0.1, n_frames=2)
        trajectory.add_trajectory(ax)
        assert_equal(len(ax.lines), 7)  # 2 * 3 axes + connection line
        assert_equal(len(ax.artists), 1)  # arrow
    finally:
        ax.remove()


def test_plot_box():
    ax = make_3d_axis(1.0)
    try:
        plot_box(ax, wireframe=False)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_box_wireframe():
    ax = make_3d_axis(1.0)
    try:
        plot_box(ax, wireframe=True)
        assert_greater_equal(len(ax.lines), 1)
    finally:
        ax.remove()


def test_plot_sphere():
    ax = make_3d_axis(1.0)
    try:
        plot_sphere(ax, wireframe=False)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_sphere_wireframe():
    ax = make_3d_axis(1.0)
    try:
        plot_sphere(ax, wireframe=True)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_cylinder():
    ax = make_3d_axis(1.0)
    try:
        plot_cylinder(ax, wireframe=False)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_cylinder_wireframe():
    ax = make_3d_axis(1.0)
    try:
        plot_cylinder(ax, wireframe=True)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_mesh():
    try:
        import trimesh
    except ImportError:
        raise SkipTest("trimesh is required for this test")
    ax = make_3d_axis(1.0)
    try:
        plot_mesh(ax, filename="test/test_data/cone.stl", wireframe=False)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_mesh_wireframe():
    try:
        import trimesh
    except ImportError:
        raise SkipTest("trimesh is required for this test")
    ax = make_3d_axis(1.0)
    try:
        plot_mesh(ax, filename="test/test_data/cone.stl", wireframe=True)
        assert_equal(len(ax.collections), 1)
    finally:
        ax.remove()


def test_plot_vector():
    ax = make_3d_axis(1.0)
    try:
        plot_vector(ax)
        assert_equal(len(ax.artists), 1)
    finally:
        ax.remove()


def test_plot_length_variable():
    ax = make_3d_axis(1.0)
    try:
        plot_length_variable(ax)
        assert_greater_equal(len(ax.lines), 1)
        assert_equal(len(ax.texts), 1)
    finally:
        ax.remove()
