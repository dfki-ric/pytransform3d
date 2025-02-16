from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr


def test_id_rot():
    """Test equivalence of constants that represent no rotation."""
    assert_array_almost_equal(pr.R_id, pr.matrix_from_axis_angle(pr.a_id))
    assert_array_almost_equal(pr.R_id, pr.matrix_from_quaternion(pr.q_id))
