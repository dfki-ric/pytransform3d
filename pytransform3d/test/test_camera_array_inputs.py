import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pytransform3d.camera import cam2sensor


@pytest.mark.parametrize("container", [list, tuple, np.asarray])
@pytest.mark.parametrize("dimensions", [3, 4])
@pytest.mark.parametrize("kappa", [0.0, 0.2, -0.01])
def test_camera_array_like_points(container, dimensions, kappa):
    points = np.array(
        [[2.0, -4, 2, 1], [-3, 6, 3, 1], [1, 1, -1, 1], [1, 1, 0, 1]]
    )[:, :dimensions]
    argument = container(points.tolist())
    before = np.asarray(argument).copy()
    result = cam2sensor(argument, 0.5, kappa)
    normalized = np.array([[1.0, -2], [-1, 2]])
    expected = (
        0.5
        * normalized
        / (1 + kappa * np.sum(normalized**2, axis=1, keepdims=True))
    )
    assert_allclose(result[:2], expected)
    assert np.isnan(result[2:]).all()
    assert result.shape == (4, 2)
    assert_array_equal(argument, before)


@pytest.mark.parametrize("container", [list, tuple, np.asarray])
def test_camera_array_like_validation(container):
    with pytest.raises(ValueError, match="dimensions"):
        cam2sensor(container([[1, 2]]), 1)
    with pytest.raises(ValueError, match="Focal length"):
        cam2sensor(container([[1, 2, 3]]), 0)
