import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pytransform3d.transform_manager import TransformManager


def translation(x):
    transform = np.eye(4)
    transform[0, 3] = x
    return transform


@pytest.mark.parametrize("json_roundtrip", [False, True])
@pytest.mark.parametrize("same_frames", [False, True])
def test_restored_state_discards_cached_paths(json_roundtrip, same_frames):
    current = TransformManager()
    current.add_transform("a", "b", translation(1))
    current.add_transform("b", "c", translation(2))
    assert_allclose(current.get_transform("a", "c"), translation(3))
    current.get_transform("c", "a")

    restored = TransformManager()
    if same_frames:
        restored.add_transform("a", "unused", translation(100))
    restored.add_transform("a", "x", translation(4))
    restored.add_transform("x", "c", translation(5))
    if same_frames:
        # Same graph size and frame membership, but different paths.
        restored.add_transform("a", "b", translation(50))
    state = restored.to_dict()
    if json_roundtrip:
        state = json.loads(json.dumps(state))
    current.set_transform_manager_state(state)
    assert_allclose(current.get_transform("a", "c"), translation(9))
    assert_allclose(current.get_transform("c", "a"), translation(-9))


@pytest.mark.parametrize(
    "graph_kind", ["source_last", "isolated_tail", "all_isolated"]
)
@pytest.mark.parametrize("json_roundtrip", [False, True])
def test_restore_preserves_sparse_graph_shape(graph_kind, json_roundtrip):
    graph = TransformManager()
    graph.add_transform("a", "b", translation(1))
    if graph_kind == "source_last":
        graph.add_transform("c", "a", translation(2))
        expected_components = 1
    elif graph_kind == "isolated_tail":
        graph.add_transform("a", "c", translation(2))
        graph.remove_transform("a", "c")
        expected_components = 2
    else:
        graph.remove_transform("a", "b")
        expected_components = 2
    state = graph.to_dict()
    if json_roundtrip:
        state = json.loads(json.dumps(state))
    restored = TransformManager.from_dict(state)
    assert restored.connections.shape == graph.connections.shape
    assert restored.connected_components() == expected_components
