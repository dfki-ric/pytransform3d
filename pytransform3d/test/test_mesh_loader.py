from pytransform3d import _mesh_loader

import pytest


def test_trimesh():
    mesh = _mesh_loader._Trimesh("test/test_data/cone.stl")
    loader_available = mesh.load()
    if not loader_available:
        pytest.skip("trimesh is required for this test")

    assert len(mesh.vertices) == 64
    assert len(mesh.triangles) == 124

    mesh.convex_hull()

    assert len(mesh.vertices) == 64


def test_trimesh_scene():
    mesh = _mesh_loader._Trimesh("test/test_data/scene.obj")
    try:
        mesh_loaded = mesh.load()
    except ImportError as e:
        if e.name == "open3d":
            pytest.skip("open3d is required for this test")
        else:
            raise e

    if not mesh_loaded:
        pytest.skip("trimesh is required for this test")

    assert mesh_loaded
    assert len(mesh.vertices) == 16
    assert len(mesh.triangles) == 24

    mesh.convex_hull()

    assert len(mesh.vertices) == 8


def test_open3d():
    mesh = _mesh_loader._Open3DMesh("test/test_data/cone.stl")
    loader_available = mesh.load()
    if not loader_available:
        pytest.skip("open3d is required for this test")

    assert len(mesh.vertices) == 295
    assert len(mesh.triangles) == 124

    o3d_mesh = mesh.get_open3d_mesh()
    assert len(o3d_mesh.vertices) == 295

    mesh.convex_hull()

    assert len(mesh.vertices) == 64


def test_trimesh_with_open3d():
    mesh = _mesh_loader._Trimesh("test/test_data/cone.stl")
    loader_available = mesh.load()
    if not loader_available:
        pytest.skip("trimesh is required for this test")
    try:
        o3d_mesh = mesh.get_open3d_mesh()
    except ImportError:
        pytest.skip("open3d is required for this test")
    assert len(o3d_mesh.vertices) == 64


def test_interface():
    try:
        mesh = _mesh_loader.load_mesh("test/test_data/cone.stl")
        assert len(mesh.triangles) == 124
    except ImportError as e:
        if e.name in ["open3d", "trimesh"]:
            pytest.skip("trimesh or open3d are required for this test")
        else:
            raise e


def test_interface_with_scene():
    try:
        mesh = _mesh_loader.load_mesh("test/test_data/scene.obj")
        assert len(mesh.triangles) == 24
    except ImportError as e:
        if e.name in ["open3d", "trimesh"]:
            pytest.skip("trimesh and open3d are required for this test")
        else:
            raise e
