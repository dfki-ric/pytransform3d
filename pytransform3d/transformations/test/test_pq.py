import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def test_check_pq():
    """Test input validation for position and orientation quaternion."""
    q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    q2 = pt.check_pq(q)
    assert_array_almost_equal(q, q2)

    q3 = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    q4 = pt.check_pq(q3)
    assert_array_almost_equal(q3, q4)
    assert len(q3) == q4.shape[0]

    A2B = np.eye(4)
    with pytest.raises(ValueError, match="position and orientation quaternion"):
        pt.check_pq(A2B)
    q5 = np.zeros(8)
    with pytest.raises(ValueError, match="position and orientation quaternion"):
        pt.check_pq(q5)


def test_pq_slerp():
    start = np.array([0.2, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0])
    end = np.array([1.0, 0.5, 0.8, 0.0, 1.0, 0.0, 0.0])
    pq_05 = pt.pq_slerp(start, end, 0.5)
    assert_array_almost_equal(pq_05, [0.6, 0.4, 0.6, 0.707107, 0.707107, 0, 0])
    pq_025 = pt.pq_slerp(start, end, 0.25)
    assert_array_almost_equal(pq_025, [0.4, 0.35, 0.5, 0.92388, 0.382683, 0, 0])
    pq_075 = pt.pq_slerp(start, end, 0.75)
    assert_array_almost_equal(pq_075, [0.8, 0.45, 0.7, 0.382683, 0.92388, 0, 0])


def test_transform_from_pq():
    """Test conversion from position and quaternion to homogeneous matrix."""
    pq = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    A2B = pt.transform_from_pq(pq)
    assert_array_almost_equal(A2B, np.eye(4))


def test_conversions_between_dual_quternion_and_pq():
    rng = np.random.default_rng(1000)
    for _ in range(5):
        pq = pr.random_vector(rng, 7)
        pq[3:] /= np.linalg.norm(pq[3:])
        dq = pt.dual_quaternion_from_pq(pq)
        pq2 = pt.pq_from_dual_quaternion(dq)
        assert_array_almost_equal(pq, pq2)
        dq2 = pt.dual_quaternion_from_pq(pq2)
        pt.assert_unit_dual_quaternion_equal(dq, dq2)
