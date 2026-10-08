import numpy as np
from numpy.testing import assert_allclose
import pytest

from pytransform3d import batch_rotations as pbr
from pytransform3d import rotations as pr
from pytransform3d import trajectories as ptr
from pytransform3d import transformations as pt


@pytest.mark.parametrize("integer_first", [False, True])
@pytest.mark.parametrize("batch_shape", [(1,), (2, 3)])
def test_batch_concatenation_preserves_mixed_dtype_values(
    integer_first, batch_shape
):
    identity = np.broadcast_to(
        np.array([1, 0, 0, 0, 0, 0, 0, 0]), batch_shape + (8,)
    )
    transform = pt.transform_from(
        pr.matrix_from_axis_angle([0, 0, 1, 0.7]), [0.5, 0.25, -0.125]
    )
    transformed = np.broadcast_to(
        pt.dual_quaternion_from_transform(transform), batch_shape + (8,)
    )
    left, right = (
        (identity, transformed) if integer_first else (transformed, identity)
    )
    assert_allclose(
        ptr.batch_concatenate_dual_quaternions(left, right), transformed
    )
    assert_allclose(
        pbr.batch_concatenate_quaternions(left[..., :4], right[..., :4]),
        transformed[..., :4],
    )
