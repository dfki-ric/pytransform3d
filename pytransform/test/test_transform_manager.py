import os
import pickle
import tempfile
import numpy as np
from pytransform.transformations import (random_transform, invert_transform,
                                         concat)
from pytransform.transform_manager import TransformManager
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises_regexp, assert_equal


def test_request_added_transform():
    """Request an added transform from the transform manager."""
    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    A2B_2 = tm.get_transform("A", "B")
    assert_array_almost_equal(A2B, A2B_2)


def test_request_inverse_transform():
    """Request an inverse transform from the transform manager."""
    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    A2B_2 = tm.get_transform("A", "B")
    assert_array_almost_equal(A2B, A2B_2)

    B2A = tm.get_transform("B", "A")
    B2A_2 = invert_transform(A2B)
    assert_array_almost_equal(B2A, B2A_2)


def test_transform_not_added():
    """Test request for transforms that have not been added."""
    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)
    C2D = random_transform(random_state)

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    tm.add_transform("C", "D", C2D)

    assert_raises_regexp(KeyError, "Unknown frame", tm.get_transform, "A", "G")
    assert_raises_regexp(KeyError, "Unknown frame", tm.get_transform, "G", "D")
    assert_raises_regexp(KeyError, "Cannot compute path", tm.get_transform,
                         "A", "D")


def test_request_concatenated_transform():
    """Request a concatenated transform from the transform manager."""
    random_state = np.random.RandomState(0)
    A2B = random_transform(random_state)
    B2C = random_transform(random_state)
    F2A = random_transform(random_state)

    tm = TransformManager()
    tm.add_transform("A", "B", A2B)
    tm.add_transform("B", "C", B2C)
    tm.add_transform("D", "E", np.eye(4))
    tm.add_transform("F", "A", F2A)

    A2C = tm.get_transform("A", "C")
    assert_array_almost_equal(A2C, concat(A2B, B2C))

    C2A = tm.get_transform("C", "A")
    assert_array_almost_equal(C2A, concat(invert_transform(B2C),
                                          invert_transform(A2B)))

    F2B = tm.get_transform("F", "B")
    assert_array_almost_equal(F2B, concat(F2A, A2B))


def test_update_transform():
    """Update an existing transform."""
    random_state = np.random.RandomState(0)
    A2B1 = random_transform(random_state)
    A2B2 = random_transform(random_state)

    tm = TransformManager()
    tm.add_transform("A", "B", A2B1)
    tm.add_transform("A", "B", A2B2)
    A2B = tm.get_transform("A", "B")

    # Hack: test depends on internal member
    assert_array_almost_equal(A2B, A2B2)
    assert_equal(len(tm.i), 1)
    assert_equal(len(tm.j), 1)


def test_pickle():
    """Test if a transform manager can be pickled."""
    random_state = np.random.RandomState(1)
    A2B1 = random_transform(random_state)
    A2B2 = random_transform(random_state)
    tm = TransformManager()
    tm.add_transform("A", "B", A2B1)
    tm.add_transform("A", "B", A2B2)
    A2B = tm.get_transform("A", "B")

    _, filename = tempfile.mkstemp(".pickle")
    try:
        pickle.dump(tm, open(filename, "wb"))
        tm2 = pickle.load(open(filename, "rb"))
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except WindowsError:
                pass  # workaround for permission problem on Windows
    A2B2 = tm2.get_transform("A", "B")
    assert_array_almost_equal(A2B, A2B2)


def test_whitelist():
    """Test correct handling of whitelists for plotting."""
    random_state = np.random.RandomState(2)
    A2B1 = random_transform(random_state)
    A2B2 = random_transform(random_state)
    tm = TransformManager()
    tm.add_transform("A", "B", A2B1)
    tm.add_transform("A", "B", A2B2)
    A2B = tm.get_transform("A", "B")

    nodes = tm._whitelisted_nodes(None)
    assert_equal(set(["A", "B"]), nodes)
    nodes = tm._whitelisted_nodes("A")
    assert_equal(set(["A"]), nodes)
    assert_raises_regexp(KeyError, "unknown nodes", tm._whitelisted_nodes, "C")
