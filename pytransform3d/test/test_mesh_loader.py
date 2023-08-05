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
    except ImportError:
        pytest.skip("trimesh or open3d are required for this test")
