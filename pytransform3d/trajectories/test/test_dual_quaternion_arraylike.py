"""Preserve the public array-like contract for trajectory dual quaternions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytransform3d import trajectories as ptr, transformations as pt


@pytest.mark.parametrize(
    "batch_operation, scalar_operation",
    [
        (ptr.batch_dq_conj, pt.dq_conj),
        (ptr.batch_dq_q_conj, pt.dq_q_conj),
        (
            ptr.screw_parameters_from_dual_quaternions,
            pt.screw_parameters_from_dual_quaternion,
        ),
        (ptr.dual_quaternions_power, pt.dual_quaternion_power),
    ],
    ids=["conjugate", "inverse", "screw_parameters", "power"],
)
@pytest.mark.parametrize("leading_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("container", ["ndarray", "list", "tuple"])
def test_dual_quaternion_helpers_accept_array_like(
    batch_operation, scalar_operation, leading_shape, container
):
    # Identity, straight translation, and rotation/translation about z.
    # Construct these independently of the batch conversion functions.
    c, s = np.cos(np.pi / 6.0), np.sin(np.pi / 6.0)
    poses = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0, 1.5],
            [c, 0.0, 0.0, s, -0.75 * s, 0.0, 0.0, 0.75 * c],
        ]
    )
    if leading_shape == ():
        dqs = poses[2].copy()
    elif leading_shape == (3,):
        dqs = poses.copy()
    else:
        dqs = np.stack([poses, poses[::-1]])
    original = dqs.copy()
    if container == "ndarray":
        # Coercion must preserve the existing read-only-array behavior.
        dqs.setflags(write=False)
        values = dqs
    elif container == "list":
        values = dqs.tolist()
    else:
        values = tuple(dqs.tolist())

    scalar_args = (
        (0.5,) if batch_operation is ptr.dual_quaternions_power else ()
    )
    references = [
        scalar_operation(dq, *scalar_args) for dq in dqs.reshape(-1, 8)
    ]
    actual = batch_operation(values, *scalar_args)
    if batch_operation is ptr.screw_parameters_from_dual_quaternions:
        for component, result in enumerate(actual):
            component_shape = (3,) if component < 2 else ()
            expected = np.asarray(
                [reference[component] for reference in references]
            ).reshape(leading_shape + component_shape)
            assert result.shape == expected.shape
            assert_allclose(result, expected)
    else:
        expected = np.asarray(references).reshape(leading_shape + (8,))
        assert actual.shape == expected.shape
        assert_allclose(actual, expected)
    assert_allclose(np.asarray(values), original)
