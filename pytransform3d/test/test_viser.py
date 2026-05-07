"""Tests for the viser visualization backend."""

import os

import numpy as np
import pytest

viser = pytest.importorskip("viser")

from pytransform3d.viser import (  # noqa: E402
    Figure,
    Artist,
    Line3D,
    PointCollection3D,
    Vector3D,
    Frame,
    Trajectory,
    Sphere,
    Box,
    Cylinder,
    Ellipsoid,
    Capsule,
    Cone,
    Plane,
    Camera,
    Graph,
)
from pytransform3d.transform_manager import TransformManager  # noqa: E402

# Sequential port counter to avoid port conflicts between tests
_PORT = [9200]


def _next_port():
    _PORT[0] += 1
    return _PORT[0]


@pytest.fixture
def fig():
    f = Figure(port=_next_port())
    yield f
    f._server.stop()


# ---------------------------------------------------------------------------
# Artist unit tests (no figure required)
# ---------------------------------------------------------------------------


def test_artist_base():
    a = Artist()
    assert a.geometries == []
    a.add_artist(None)  # should not raise
    a.remove()  # should not raise


def test_line3d_init():
    P = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
    line = Line3D(P)
    assert line._handle is None


def test_line3d_single_color():
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    line = Line3D(P, c=(1, 0, 0))
    assert line._handle is None


def test_line3d_per_segment_colors():
    P = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    c = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    line = Line3D(P, c=c)
    assert line._handle is None


def test_point_collection_init():
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    pc = PointCollection3D(P, s=0.1, c=(0, 1, 0))
    assert pc._handle is None


def test_vector3d_init():
    v = Vector3D(start=[0, 0, 0], direction=[1, 0, 0], c=(0, 0, 1))
    assert v._handle is None


def test_vector3d_zero_direction():
    v = Vector3D(start=[0, 0, 0], direction=[0, 0, 0])
    assert v._handle is None


def test_frame_init():
    f = Frame(np.eye(4), s=1.0)
    assert f._handle is None


def test_frame_with_label():
    f = Frame(np.eye(4), label="test", s=0.5)
    assert f.label == "test"


def test_trajectory_init():
    H = np.tile(np.eye(4), (10, 1, 1))
    traj = Trajectory(H, n_frames=5, s=0.2)
    assert len(traj.key_frames) == 5


def test_sphere_init():
    s = Sphere(radius=0.5, c=(1, 0, 0))
    assert s._handle is None


def test_sphere_no_color():
    s = Sphere(radius=1.0)
    assert s.c is None


def test_box_init():
    b = Box(size=[1, 2, 3], c=(0, 0, 1))
    assert b._handle is None


def test_cylinder_init():
    c = Cylinder(length=2.0, radius=0.5, resolution=16)
    assert c._handle is None


def test_cylinder_split_ignored():
    # split parameter accepted for API compatibility but not used
    c = Cylinder(length=1.0, radius=0.5, split=8)
    assert c._handle is None


def test_ellipsoid_init():
    e = Ellipsoid(radii=[1, 2, 3])
    assert e._handle is None


def test_capsule_init():
    cap = Capsule(height=1.0, radius=0.3)
    assert cap._handle is None


def test_cone_init():
    co = Cone(height=1.0, radius=0.5)
    assert co._handle is None


def test_plane_init_with_d():
    pl = Plane(normal=[0, 0, 1], d=0.5)
    assert pl.d == 0.5


def test_plane_init_with_point():
    pl = Plane(normal=[0, 0, 1], point_in_plane=[0, 0, 1])
    assert pl.point_in_plane is not None


def test_plane_missing_both_raises():
    pl = Plane(normal=[0, 0, 1])
    with pytest.raises(ValueError, match="Either 'd' or 'point_in_plane'"):
        pl._resolve_point(pl.normal, pl.d, pl.point_in_plane)


def test_camera_init():
    M = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=float)
    cam = Camera(M)
    assert cam._handle is None


def test_camera_init_with_cam2world():
    M = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=float)
    cam2world = np.eye(4)
    cam = Camera(M, cam2world=cam2world)
    assert cam._handle is None


# ---------------------------------------------------------------------------
# Figure integration tests
# ---------------------------------------------------------------------------


def test_figure_creation(fig):
    assert fig._server is not None
    assert fig._object_count == 0


def test_figure_show_prints_url(fig, capsys):
    fig.show()
    captured = capsys.readouterr()
    assert "localhost" in captured.out


def test_figure_save_image(fig):
    pytest.importorskip("playwright")
    pytest.importorskip("imageio")
    fig.save_image("/tmp/test_viser_save.jpg")
    assert os.path.exists("/tmp/test_viser_save.jpg")


def test_figure_set_line_width_warns(fig):
    with pytest.warns(UserWarning):
        fig.set_line_width(2.0)


def test_plot_line(fig):
    P = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
    line = fig.plot(P)
    assert isinstance(line, Line3D)
    assert line._handle is not None


def test_plot_line_single_point_no_handle(fig):
    P = np.array([[0, 0, 0]], dtype=float)
    line = fig.plot(P)
    assert isinstance(line, Line3D)
    assert line._handle is None  # not enough points


def test_plot_line_set_data(fig):
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    line = fig.plot(P)
    P2 = np.array([[0, 0, 0], [0, 1, 0]], dtype=float)
    line.set_data(P2)  # should not raise


def test_scatter(fig):
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    pc = fig.scatter(P, s=0.05, c=(1, 0, 0))
    assert isinstance(pc, PointCollection3D)
    assert pc._handle is not None


def test_scatter_no_color(fig):
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    pc = fig.scatter(P)
    assert isinstance(pc, PointCollection3D)


def test_scatter_per_point_color(fig):
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    c = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
    pc = fig.scatter(P, c=c)
    assert isinstance(pc, PointCollection3D)


def test_scatter_set_data(fig):
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    pc = fig.scatter(P)
    pc.set_data(np.array([[0, 0, 0], [0, 1, 0]], dtype=float))


def test_plot_vector(fig):
    v = fig.plot_vector([0, 0, 0], [1, 0, 0], c=(0, 1, 0))
    assert isinstance(v, Vector3D)
    assert v._handle is not None


def test_plot_vector_zero_length(fig):
    v = fig.plot_vector([0, 0, 0], [0, 0, 0])
    assert isinstance(v, Vector3D)
    assert v._handle is None


def test_plot_vector_set_data(fig):
    v = fig.plot_vector([0, 0, 0], [1, 0, 0])
    v.set_data([0, 0, 0], [0, 1, 0])
    assert v._handle is not None


def test_plot_basis(fig):
    f = fig.plot_basis()
    assert isinstance(f, Frame)
    assert f._handle is not None


def test_plot_basis_with_rotation(fig):
    R = np.eye(3)
    f = fig.plot_basis(R=R, p=[1, 0, 0], s=0.5)
    assert isinstance(f, Frame)


def test_plot_transform_identity(fig):
    f = fig.plot_transform(np.eye(4))
    assert isinstance(f, Frame)
    assert f._handle is not None


def test_plot_transform_none(fig):
    f = fig.plot_transform()
    assert isinstance(f, Frame)


def test_plot_transform_set_data(fig):
    f = fig.plot_transform(np.eye(4))
    f.set_data(np.eye(4))


def test_plot_transform_with_name(fig):
    f = fig.plot_transform(np.eye(4), name="world")
    assert isinstance(f, Frame)


def test_plot_trajectory(fig):
    pqs = np.zeros((10, 7))
    pqs[:, 0] = np.linspace(0, 1, 10)
    pqs[:, 3] = 1.0
    traj = fig.plot_trajectory(pqs, n_frames=5, s=0.2)
    assert isinstance(traj, Trajectory)
    assert traj.line._handle is not None
    assert len(traj.key_frames) == 5


def test_plot_trajectory_set_data(fig):
    pqs = np.zeros((10, 7))
    pqs[:, 0] = np.linspace(0, 1, 10)
    pqs[:, 3] = 1.0
    traj = fig.plot_trajectory(pqs, n_frames=3)
    pqs2 = np.zeros((10, 7))
    pqs2[:, 1] = np.linspace(0, 1, 10)
    pqs2[:, 3] = 1.0
    from pytransform3d.trajectories import transforms_from_pqs

    traj.set_data(transforms_from_pqs(pqs2))


def test_plot_sphere(fig):
    s = fig.plot_sphere(radius=0.5, c=(1, 0, 0))
    assert isinstance(s, Sphere)
    assert s._handle is not None


def test_plot_sphere_set_data(fig):
    s = fig.plot_sphere(radius=0.5)
    s.set_data(np.eye(4))


def test_plot_box(fig):
    b = fig.plot_box(size=[1, 2, 3], c=(0, 0, 1))
    assert isinstance(b, Box)
    assert b._handle is not None


def test_plot_box_set_data(fig):
    b = fig.plot_box()
    b.set_data(np.eye(4))


def test_plot_cylinder(fig):
    c = fig.plot_cylinder(length=2.0, radius=0.5)
    assert isinstance(c, Cylinder)
    assert c._handle is not None


def test_plot_cylinder_set_data(fig):
    c = fig.plot_cylinder()
    c.set_data(np.eye(4))


def test_plot_ellipsoid(fig):
    e = fig.plot_ellipsoid(radii=[1, 2, 3], c=(1, 0, 1))
    assert isinstance(e, Ellipsoid)
    assert e._handle is not None


def test_plot_ellipsoid_set_data(fig):
    e = fig.plot_ellipsoid(radii=[1, 1, 1])
    e.set_data(np.eye(4))


def test_plot_capsule(fig):
    cap = fig.plot_capsule(height=1.0, radius=0.3)
    assert isinstance(cap, Capsule)
    assert cap._handle is not None


def test_plot_capsule_set_data(fig):
    cap = fig.plot_capsule()
    cap.set_data(np.eye(4))


def test_plot_cone(fig):
    co = fig.plot_cone(height=1.0, radius=0.5, c=(1, 1, 0))
    assert isinstance(co, Cone)
    assert co._handle is not None


def test_plot_cone_set_data(fig):
    co = fig.plot_cone()
    co.set_data(np.eye(4))


def test_plot_plane_with_d(fig):
    pl = fig.plot_plane(normal=[0, 0, 1], d=0.0)
    assert isinstance(pl, Plane)
    assert pl._handle is not None


def test_plot_plane_with_point_in_plane(fig):
    pl = fig.plot_plane(normal=[0, 0, 1], point_in_plane=[0, 0, 0])
    assert isinstance(pl, Plane)


def test_plot_plane_set_data(fig):
    pl = fig.plot_plane(normal=[0, 0, 1], d=0.0)
    pl.set_data(normal=[1, 0, 0], d=0.0)


def test_plot_plane_set_data_raises_without_definition(fig):
    pl = fig.plot_plane(normal=[0, 0, 1], d=0.0)
    with pytest.raises(ValueError):
        pl.set_data(normal=[0, 0, 1])


def test_plot_camera(fig):
    M = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=float)
    cam = fig.plot_camera(M=M)
    assert isinstance(cam, Camera)
    assert cam._handle is not None


def test_plot_camera_with_cam2world(fig):
    M = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=float)
    cam = fig.plot_camera(M=M, cam2world=np.eye(4), virtual_image_distance=0.5)
    assert isinstance(cam, Camera)


def test_plot_camera_set_data(fig):
    M = np.array([[800, 0, 960], [0, 800, 540], [0, 0, 1]], dtype=float)
    cam = fig.plot_camera(M=M)
    cam.set_data(cam2world=np.eye(4))


def test_plot_graph_empty(fig):
    tm = TransformManager()
    tm.add_transform("A", "world", np.eye(4))
    g = fig.plot_graph(tm, "world")
    assert isinstance(g, Graph)


def test_plot_graph_with_frames(fig):
    tm = TransformManager()
    tm.add_transform("A", "world", np.eye(4))
    tm.add_transform("B", "world", np.eye(4))
    g = fig.plot_graph(tm, "world", show_frames=True, s=0.5)
    assert isinstance(g, Graph)
    assert len(g.frames) == 3  # world, A, B


def test_plot_graph_with_connections(fig):
    tm = TransformManager()
    tm.add_transform("A", "world", np.eye(4))
    g = fig.plot_graph(tm, "world", show_connections=True)
    assert isinstance(g, Graph)
    assert len(g.connections) == 1


def test_plot_graph_set_data(fig):
    tm = TransformManager()
    tm.add_transform("A", "world", np.eye(4))
    g = fig.plot_graph(tm, "world", show_frames=True)
    g.set_data()


def test_plot_graph_unknown_frame_raises(fig):
    tm = TransformManager()
    tm.add_transform("A", "world", np.eye(4))
    with pytest.raises(KeyError):
        fig.plot_graph(tm, "nonexistent")


def test_remove_artist_line(fig):
    P = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    line = fig.plot(P)
    fig.remove_artist(line)
    assert line._handle is None


def test_remove_artist_sphere(fig):
    s = fig.plot_sphere()
    fig.remove_artist(s)
    assert s._handle is None


def test_remove_artist_graph(fig):
    tm = TransformManager()
    tm.add_transform("A", "world", np.eye(4))
    g = fig.plot_graph(tm, "world", show_frames=True, show_connections=True)
    fig.remove_artist(g)


def test_remove_artist_trajectory(fig):
    pqs = np.zeros((5, 7))
    pqs[:, 3] = 1.0
    traj = fig.plot_trajectory(pqs, n_frames=2)
    fig.remove_artist(traj)


def test_next_name_increments(fig):
    n1 = fig._next_name("test")
    n2 = fig._next_name("test")
    assert n1 != n2
