import numpy as np
from pytransform.camera import cam2sensor
from nose.tools import assert_raises_regexp, assert_false
from numpy.testing import assert_array_almost_equal


def test_cam2sensor_wrong_dimensions():
    P_cam = np.ones((1, 2))
    assert_raises_regexp(ValueError, "3- or 4-dimensional points",
                         cam2sensor, P_cam, 0.1)
    P_cam = np.ones((1, 5))
    assert_raises_regexp(ValueError, "3- or 4-dimensional points",
                         cam2sensor, P_cam, 0.1)


def test_cam2sensor_wrong_focal_length():
    P_cam = np.ones((1, 3))
    assert_raises_regexp(ValueError, "must be greater than 0",
                         cam2sensor, P_cam, 0.0)


def test_cam2sensor_points_behind_camera():
    P_cam = np.ones((1, 3))
    P_cam[0, 2] = -1.0
    P_sensor = cam2sensor(P_cam, 0.1)
    assert_false(np.any(np.isfinite(P_sensor)))


def test_cam2sensor_projection():
    P_cam = np.array([[-0.56776587, 0.03855521, 0.81618344, 1.0]])
    P_sensor = cam2sensor(P_cam, 0.0036)
    assert_array_almost_equal(P_sensor, np.array([[-0.00250429, 0.00017006]]))
