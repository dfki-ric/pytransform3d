import os
import pickle
import warnings
import platform
import tempfile
import numpy as np
from pytransform3d.rotations import (
    q_id, active_matrix_from_intrinsic_euler_xyz, random_quaternion)
from pytransform3d.transformations import (
    random_transform, invert_transform, concat, transform_from_pq,
    transform_from)
from pytransform3d.transform_manager import TransformManager
from pytransform3d import transform_manager
from numpy.testing import assert_array_almost_equal
from nose.tools import (assert_raises_regexp, assert_equal, assert_true,
                        assert_false)
from nose import SkipTest


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


def test_has_frame():
    """Check if frames have been registered with transform."""
    tm = TransformManager()
    tm.add_transform("A", "B", np.eye(4))
    assert_true(tm.has_frame("A"))
    assert_true(tm.has_frame("B"))
    assert_false(tm.has_frame("C"))


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
    A2B = random_transform(random_state)
    tm = TransformManager()
    tm.add_transform("A", "B", A2B)

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
    A2B = random_transform(random_state)
    tm = TransformManager()
    tm.add_transform("A", "B", A2B)

    nodes = tm._whitelisted_nodes(None)
    assert_equal({"A", "B"}, nodes)
    nodes = tm._whitelisted_nodes(["A"])
    assert_equal({"A"}, nodes)
    assert_raises_regexp(KeyError, "unknown nodes", tm._whitelisted_nodes, "C")


def test_check_consistency():
    """Test correct detection of inconsistent graphs."""
    random_state = np.random.RandomState(2)

    tm = TransformManager()

    A2B = random_transform(random_state)
    tm.add_transform("A", "B", A2B)
    B2A = random_transform(random_state)
    tm.add_transform("B", "A", B2A)
    assert_false(tm.check_consistency())

    tm = TransformManager()

    A2B = random_transform(random_state)
    tm.add_transform("A", "B", A2B)
    assert_true(tm.check_consistency())

    C2D = random_transform(random_state)
    tm.add_transform("C", "D", C2D)
    assert_true(tm.check_consistency())

    B2C = random_transform(random_state)
    tm.add_transform("B", "C", B2C)
    assert_true(tm.check_consistency())

    A2D_over_path = tm.get_transform("A", "D")

    A2D = random_transform(random_state)
    tm.add_transform("A", "D", A2D)
    assert_false(tm.check_consistency())

    tm.add_transform("A", "D", A2D_over_path)
    assert_true(tm.check_consistency())


def test_connected_components():
    """Test computation of connected components in the graph."""
    tm = TransformManager()
    tm.add_transform("A", "B", np.eye(4))
    assert_equal(tm.connected_components(), 1)
    tm.add_transform("D", "E", np.eye(4))
    assert_equal(tm.connected_components(), 2)
    tm.add_transform("B", "C", np.eye(4))
    assert_equal(tm.connected_components(), 2)
    tm.add_transform("D", "C", np.eye(4))
    assert_equal(tm.connected_components(), 1)


def test_png_export():
    """Test if the graph can be exported to PNG."""
    random_state = np.random.RandomState(0)

    ee2robot = transform_from_pq(
        np.hstack((np.array([0.4, -0.3, 0.5]),
                   random_quaternion(random_state))))
    cam2robot = transform_from_pq(
        np.hstack((np.array([0.0, 0.0, 0.8]), q_id)))
    object2cam = transform_from(
        active_matrix_from_intrinsic_euler_xyz(np.array([0.0, 0.0, 0.5])),
        np.array([0.5, 0.1, 0.1]))

    tm = TransformManager()
    tm.add_transform("end-effector", "robot", ee2robot)
    tm.add_transform("camera", "robot", cam2robot)
    tm.add_transform("object", "camera", object2cam)

    _, filename = tempfile.mkstemp(".png")
    try:
        tm.write_png(filename)
        assert_true(os.path.exists(filename))
    except ImportError:
        raise SkipTest("pydot is required for this test")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except WindowsError:
                pass  # workaround for permission problem on Windows


def test_png_export_without_pydot_fails():
    """Test graph export without pydot."""
    pydot_available = transform_manager.PYDOT_AVAILABLE
    tm = TransformManager()
    try:
        transform_manager.PYDOT_AVAILABLE = False
        assert_raises_regexp(
            ImportError, "pydot must be installed to use this feature.",
            tm.write_png, "bla")
    finally:
        transform_manager.PYDOT_AVAILABLE = pydot_available


def test_deactivate_transform_manager_precision_error():
    A2B = np.eye(4)
    A2B[0, 0] = 2.0
    A2B[3, 0] = 3.0
    tm = TransformManager()
    assert_raises_regexp(
        ValueError, "Expected rotation matrix",
        tm.add_transform, "A", "B", A2B)

    if int(platform.python_version()[0]) == 2:
        # Python 2 seems to incorrectly suppress some warnings, not sure why
        n_expected_warnings = 7
    else:
        n_expected_warnings = 9
    try:
        warnings.filterwarnings("always", category=UserWarning)
        with warnings.catch_warnings(record=True) as w:
            tm = TransformManager(strict_check=False)
            tm.add_transform("A", "B", A2B)
            tm.add_transform("B", "C", np.eye(4))
            tm.get_transform("C", "A")
            assert_equal(len(w), n_expected_warnings)
    finally:
        warnings.filterwarnings("default", category=UserWarning)


def test_deactivate_checks():
    tm = TransformManager(check=False)
    tm.add_transform("A", "B", np.zeros((4, 4)))
    tm.add_transform("B", "C", np.zeros((4, 4)))
    A2B = tm.get_transform("A", "C")
    assert_array_almost_equal(A2B, np.zeros((4, 4)))


def test_remove_transform():
    tm = TransformManager()
    tm.add_transform("A", "B", np.eye(4))
    tm.add_transform("C", "D", np.eye(4))

    assert_raises_regexp(
        KeyError, "Cannot compute path", tm.get_transform, "A", "D")

    tm.add_transform("B", "C", np.eye(4))
    tm.get_transform("A", "C")

    tm.remove_transform("B", "C")
    tm.remove_transform("B", "C")  # nothing should happen
    assert_raises_regexp(
        KeyError, "Cannot compute path", tm.get_transform, "A", "D")
    tm.get_transform("B", "A")
    tm.get_transform("D", "C")


def test_from_to_dict():
    random_state = np.random.RandomState(2323)
    tm = TransformManager()
    A2B = random_transform(random_state)
    tm.add_transform("A", "B", A2B)
    B2C = random_transform(random_state)
    tm.add_transform("B", "C", B2C)
    C2D = random_transform(random_state)
    tm.add_transform("C", "D", C2D)

    tm_dict = tm.to_dict()
    tm2 = TransformManager.from_dict(tm_dict)

    assert_array_almost_equal(tm.get_transform("D", "A"),
                              tm2.get_transform("D", "A"))
