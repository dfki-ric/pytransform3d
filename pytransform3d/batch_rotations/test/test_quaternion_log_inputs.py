import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from pytransform3d.batch_rotations import axis_angles_from_quaternions


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int64, bool])
@pytest.mark.parametrize("shape", [(), (4,), (2, 2)])
def test_integral_unit_quaternion_log(dtype, shape):
    quaternions = np.eye(4, dtype=dtype)
    if not shape:
        quaternions = quaternions[1]
    else:
        quaternions = quaternions.reshape(shape + (4,))
    quaternions.flags.writeable = False
    original = quaternions.copy()
    result = axis_angles_from_quaternions(quaternions)
    assert result.shape == quaternions.shape
    expected_angles = np.array([0, np.pi, np.pi, np.pi])
    if not shape:
        assert_allclose(result, [1, 0, 0, np.pi])
    else:
        assert_allclose(result[..., 3], expected_angles.reshape(shape))
    rotvec = (result[..., :3] * result[..., 3:]).reshape(-1, 3)
    q_xyzw = quaternions.reshape(-1, 4)[:, [1, 2, 3, 0]]
    assert_allclose(
        Rotation.from_rotvec(rotvec).as_matrix(),
        Rotation.from_quat(q_xyzw).as_matrix(),
        atol=1e-15,
    )
    assert_array_equal(quaternions, original)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("batched", [False, True])
def test_array_like_quaternion_log(container, batched):
    quaternions = np.array([[0.5, 0.5, 0.5, 0.5], [0, 0, 1, 0]])
    if not batched:
        quaternions = quaternions[0]
    value = container(quaternions.tolist())
    assert_allclose(
        axis_angles_from_quaternions(value),
        axis_angles_from_quaternions(quaternions),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_quaternion_log_control(dtype):
    q = np.array([[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0]], dtype=dtype)
    original = q.copy()
    result = axis_angles_from_quaternions(q)
    assert result.dtype == q.dtype
    assert_allclose(
        result[0], [1 / np.sqrt(3)] * 3 + [2 * np.pi / 3], rtol=1e-6
    )
    assert_allclose(result[1], [1, 0, 0, 0])
    assert_array_equal(q, original)


def test_empty_quaternion_log():
    assert axis_angles_from_quaternions(
        np.empty((2, 0, 4), dtype=int)
    ).shape == (2, 0, 4)
