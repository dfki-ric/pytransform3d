"""Tests for the swing-twist decomposition of a rotation."""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import pytransform3d.rotations as pr

# A representative selection of twist axes: the canonical basis vectors and a
# few arbitrary directions (deliberately not normalized).
AXES = [
    pr.unitx,
    pr.unity,
    pr.unitz,
    -pr.unitz,
    np.array([1.0, 1.0, 1.0]),
    np.array([0.2, -1.0, 0.5]),
    np.array([-3.0, 0.0, 4.0]),
]


def _random_quaternions(n, seed):
    rng = np.random.default_rng(seed)
    return [pr.random_quaternion(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Fundamental identity: q = swing * twist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", _random_quaternions(30, seed=42))
@pytest.mark.parametrize("axis", AXES)
def test_reconstruction(q, axis):
    swing, twist = pr.swing_twist_decomposition(q, axis)
    # q == swing * twist (up to sign, which represents the same rotation)
    assert pr.quaternion_dist(
        pr.concatenate_quaternions(swing, twist), q
    ) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("q", _random_quaternions(30, seed=42))
@pytest.mark.parametrize("axis", AXES)
def test_composition_roundtrip(q, axis):
    # swing_twist_composition is the inverse of swing_twist_decomposition;
    # it recovers the original rotation (up to sign).
    swing, twist = pr.swing_twist_decomposition(q, axis)
    q_reconstructed = pr.swing_twist_composition(swing, twist)
    assert pr.quaternion_dist(q_reconstructed, q) == pytest.approx(
        0.0, abs=1e-10
    )


def test_composition_matches_concatenation():
    # swing_twist_composition applies the twist first, then the swing, i.e.
    # it is exactly concatenate_quaternions(swing, twist).
    swing = pr.quaternion_from_axis_angle([0.0, 1.0, 0.0, 0.7])
    twist = pr.quaternion_from_axis_angle([1.0, 0.0, 0.0, 0.3])
    assert_array_almost_equal(
        pr.swing_twist_composition(swing, twist),
        pr.concatenate_quaternions(swing, twist),
    )


@pytest.mark.parametrize("q", _random_quaternions(10, seed=101))
@pytest.mark.parametrize("axis", AXES)
def test_reconstruction_as_matrix_product(q, axis):
    # The decomposition must also hold for the equivalent rotation matrices:
    # R = R_swing @ R_twist.
    swing, twist = pr.swing_twist_decomposition(q, axis)
    R = pr.matrix_from_quaternion(q)
    R_swing = pr.matrix_from_quaternion(swing)
    R_twist = pr.matrix_from_quaternion(twist)
    assert_array_almost_equal(R, R_swing.dot(R_twist))


@pytest.mark.parametrize("q", _random_quaternions(30, seed=7))
@pytest.mark.parametrize("axis", AXES)
def test_outputs_are_unit_quaternions(q, axis):
    swing, twist = pr.swing_twist_decomposition(q, axis)
    assert np.linalg.norm(swing) == pytest.approx(1.0)
    assert np.linalg.norm(twist) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Geometric properties of swing and twist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", _random_quaternions(30, seed=3))
@pytest.mark.parametrize("axis", AXES)
def test_twist_axis_parallel_to_given_axis(q, axis):
    _, twist = pr.swing_twist_decomposition(q, axis)
    # the vector part of the twist is parallel to the (normalized) twist axis
    assert_array_almost_equal(
        np.cross(twist[1:], pr.norm_vector(axis)), np.zeros(3)
    )


@pytest.mark.parametrize("q", _random_quaternions(30, seed=5))
@pytest.mark.parametrize("axis", AXES)
def test_swing_axis_orthogonal_to_given_axis(q, axis):
    swing, _ = pr.swing_twist_decomposition(q, axis)
    # the rotation axis of the swing is orthogonal to the twist axis
    assert np.dot(swing[1:], pr.norm_vector(axis)) == pytest.approx(
        0.0, abs=1e-10
    )


@pytest.mark.parametrize("q", _random_quaternions(30, seed=9))
@pytest.mark.parametrize("axis", AXES)
def test_twist_leaves_axis_invariant(q, axis):
    # A rotation about the axis does not move the axis itself.
    _, twist = pr.swing_twist_decomposition(q, axis)
    axis = pr.norm_vector(axis)
    assert_array_almost_equal(pr.q_prod_vector(twist, axis), axis)


@pytest.mark.parametrize("q", _random_quaternions(30, seed=11))
@pytest.mark.parametrize("axis", AXES)
def test_swing_maps_axis_like_full_rotation(q, axis):
    # Because the twist fixes the axis, the swing must move the axis exactly
    # like the full rotation does.
    swing, _ = pr.swing_twist_decomposition(q, axis)
    axis = pr.norm_vector(axis)
    assert_array_almost_equal(
        pr.q_prod_vector(swing, axis), pr.q_prod_vector(q, axis)
    )


# ---------------------------------------------------------------------------
# Special cases: identity, pure twist, pure swing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", AXES)
def test_identity_rotation(axis):
    swing, twist = pr.swing_twist_decomposition(pr.q_id, axis)
    pr.assert_quaternion_equal(swing, pr.q_id)
    pr.assert_quaternion_equal(twist, pr.q_id)


@pytest.mark.parametrize("axis", AXES)
@pytest.mark.parametrize("angle", [-2.5, -0.9, 0.3, 1.7, 3.0])
def test_pure_twist(axis, angle):
    # a rotation about the axis has no swing
    q = pr.quaternion_from_axis_angle(
        np.hstack((pr.norm_vector(axis), [angle]))
    )
    swing, twist = pr.swing_twist_decomposition(q, axis)
    pr.assert_quaternion_equal(swing, pr.q_id)
    pr.assert_quaternion_equal(twist, q)


@pytest.mark.parametrize("angle", [-2.5, -0.9, 0.3, 1.7, 3.0])
def test_pure_swing(angle):
    # a rotation about an orthogonal axis has no twist
    axis = pr.unitz
    q = pr.quaternion_from_axis_angle(np.hstack((pr.unitx, [angle])))
    swing, twist = pr.swing_twist_decomposition(q, axis)
    pr.assert_quaternion_equal(twist, pr.q_id)
    pr.assert_quaternion_equal(swing, q)


@pytest.mark.parametrize("twist_angle", [-2.0, 0.4, 1.1, 2.8])
@pytest.mark.parametrize("swing_angle", [0.2, 0.6, 1.5])
def test_recover_known_components(twist_angle, swing_angle):
    axis = pr.norm_vector(np.array([0.2, -1.0, 0.5]))
    twist_true = pr.quaternion_from_axis_angle(np.hstack((axis, [twist_angle])))
    swing_axis = pr.norm_vector(pr.perpendicular_to_vectors(axis, pr.unitx))
    swing_true = pr.quaternion_from_axis_angle(
        np.hstack((swing_axis, [swing_angle]))
    )
    q = pr.concatenate_quaternions(swing_true, twist_true)

    swing, twist = pr.swing_twist_decomposition(q, axis)
    assert pr.quaternion_dist(twist, twist_true) == pytest.approx(
        0.0, abs=1e-10
    )
    assert pr.quaternion_dist(swing, swing_true) == pytest.approx(
        0.0, abs=1e-10
    )


# ---------------------------------------------------------------------------
# Idempotency / stability of the decomposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", _random_quaternions(15, seed=13))
@pytest.mark.parametrize("axis", AXES)
def test_decomposing_twist_again_is_stable(q, axis):
    # Decomposing the twist about the same axis returns the twist unchanged
    # and an identity swing.
    _, twist = pr.swing_twist_decomposition(q, axis)
    swing2, twist2 = pr.swing_twist_decomposition(twist, axis)
    pr.assert_quaternion_equal(swing2, pr.q_id)
    pr.assert_quaternion_equal(twist2, twist)


@pytest.mark.parametrize("q", _random_quaternions(15, seed=17))
@pytest.mark.parametrize("axis", AXES)
def test_swing_has_no_twist_component(q, axis):
    # Decomposing the swing about the same axis yields an identity twist and
    # returns the swing unchanged.
    swing, _ = pr.swing_twist_decomposition(q, axis)
    swing2, twist2 = pr.swing_twist_decomposition(swing, axis)
    assert pr.quaternion_dist(twist2, pr.q_id) == pytest.approx(0.0, abs=1e-10)
    pr.assert_quaternion_equal(swing2, swing)


# ---------------------------------------------------------------------------
# Double cover: q and -q represent the same rotation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", _random_quaternions(15, seed=19))
@pytest.mark.parametrize("axis", AXES)
def test_double_cover(q, axis):
    swing, twist = pr.swing_twist_decomposition(q, axis)
    swing_neg, twist_neg = pr.swing_twist_decomposition(-q, axis)
    # -q is the same rotation, so both components describe the same rotations
    assert pr.quaternion_dist(swing_neg, swing) == pytest.approx(0.0, abs=1e-10)
    assert pr.quaternion_dist(twist_neg, twist) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Hemisphere canonicalization of the twist
#
# The twist is forced into the canonical hemisphere (non-negative scalar
# part), so that its rotation angle about the axis is in [-pi, pi] and the
# result is independent of the sign of the input quaternion. random_quaternion
# produces both signs of the scalar part, so these tests exercise the flip.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", _random_quaternions(50, seed=23))
@pytest.mark.parametrize("axis", AXES)
def test_twist_is_in_canonical_hemisphere(q, axis):
    # The scalar part of the twist must be non-negative.
    _, twist = pr.swing_twist_decomposition(q, axis)
    assert twist[0] >= 0.0


@pytest.mark.parametrize("q", _random_quaternions(50, seed=31))
@pytest.mark.parametrize("axis", AXES)
def test_twist_is_identical_for_q_and_negated_q(q, axis):
    # Thanks to the canonicalization, the twist is not just the same rotation
    # for q and -q, it is the exact same quaternion (no sign ambiguity).
    _, twist = pr.swing_twist_decomposition(q, axis)
    _, twist_neg = pr.swing_twist_decomposition(-q, axis)
    assert_array_almost_equal(twist, twist_neg)


def test_negative_scalar_input_triggers_flip():
    # Hand-built quaternion with a negative scalar part and a non-zero
    # projection onto the axis: this must be flipped into the canonical
    # hemisphere while still reconstructing the original rotation.
    axis = pr.unitz
    q = pr.norm_vector(np.array([-0.3, 0.4, 0.1, 0.2]))
    swing, twist = pr.swing_twist_decomposition(q, axis)
    assert twist[0] > 0.0
    assert pr.quaternion_dist(
        pr.concatenate_quaternions(swing, twist), q
    ) == pytest.approx(0.0, abs=1e-10)


def test_negated_pure_twist_is_canonicalized():
    # A pure twist whose quaternion has a negative scalar part is the same
    # rotation as its canonical counterpart; the decomposition returns the
    # canonical form (non-negative scalar) and no swing.
    axis = pr.norm_vector(np.array([1.0, 2.0, -1.0]))
    q_canonical = pr.quaternion_from_axis_angle(np.hstack((axis, [0.9])))
    q = -q_canonical  # same rotation, negative scalar part
    assert q[0] < 0.0

    swing, twist = pr.swing_twist_decomposition(q, axis)
    assert twist[0] >= 0.0
    pr.assert_quaternion_equal(swing, pr.q_id)
    assert pr.quaternion_dist(twist, q_canonical) == pytest.approx(
        0.0, abs=1e-10
    )


def test_twist_angle_within_pi():
    # A non-negative scalar part is equivalent to a twist angle in [-pi, pi].
    rng = np.random.default_rng(37)
    for _ in range(200):
        q = pr.random_quaternion(rng)
        axis = pr.norm_vector(rng.normal(size=3))
        _, twist = pr.swing_twist_decomposition(q, axis)
        angle = 2.0 * np.arctan2(np.linalg.norm(twist[1:]), twist[0])
        assert -np.pi - 1e-12 <= angle <= np.pi + 1e-12


# ---------------------------------------------------------------------------
# Numerical edge cases and singularities
# ---------------------------------------------------------------------------


def test_pi_rotation_about_twist_axis_is_not_singular():
    # A rotation by pi *about* the twist axis is a well defined pure twist.
    axis = pr.unitz
    q = pr.quaternion_from_axis_angle(np.hstack((axis, [np.pi])))
    swing, twist = pr.swing_twist_decomposition(q, axis)
    pr.assert_quaternion_equal(swing, pr.q_id)
    pr.assert_quaternion_equal(twist, q)


@pytest.mark.parametrize("orthogonal_axis", [pr.unitx, pr.unity])
def test_pi_rotation_orthogonal_singularity(orthogonal_axis):
    # rotation by pi about an axis orthogonal to the twist axis: the twist is
    # undefined and falls back to the identity, the swing captures everything.
    axis = pr.unitz
    q = pr.quaternion_from_axis_angle(np.hstack((orthogonal_axis, [np.pi])))
    swing, twist = pr.swing_twist_decomposition(q, axis)
    pr.assert_quaternion_equal(twist, pr.q_id)
    assert pr.quaternion_dist(
        pr.concatenate_quaternions(swing, twist), q
    ) == pytest.approx(0.0, abs=1e-10)


def test_near_pi_orthogonal_still_reconstructs():
    # Just short of the singularity the decomposition must remain accurate.
    axis = pr.unitz
    q = pr.quaternion_from_axis_angle(np.hstack((pr.unitx, [np.pi - 1e-6])))
    swing, twist = pr.swing_twist_decomposition(q, axis)
    assert np.linalg.norm(twist) == pytest.approx(1.0)
    assert pr.quaternion_dist(
        pr.concatenate_quaternions(swing, twist), q
    ) == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("axis", AXES)
def test_tiny_rotation_near_identity(axis):
    q = pr.quaternion_from_axis_angle(
        np.hstack((pr.norm_vector(np.array([1.0, -2.0, 3.0])), [1e-9]))
    )
    swing, twist = pr.swing_twist_decomposition(q, axis)
    pr.assert_quaternion_equal(swing, pr.q_id)
    pr.assert_quaternion_equal(twist, pr.q_id)


# ---------------------------------------------------------------------------
# Input handling: axis scaling, array-likes, non-unit quaternions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-3, 0.5, 1.0, 7.0, 1000.0])
def test_axis_scale_invariance(scale):
    q = pr.random_quaternion(np.random.default_rng(8))
    unit_axis = pr.norm_vector(np.array([0.0, 1.0, 1.0]))
    swing_ref, twist_ref = pr.swing_twist_decomposition(q, unit_axis)
    swing, twist = pr.swing_twist_decomposition(q, scale * unit_axis)
    assert_array_almost_equal(swing, swing_ref)
    assert_array_almost_equal(twist, twist_ref)


# Extreme scales, including the subnormal range where a naive
# sqrt(sum-of-squares) normalization would underflow or lose precision. The
# decomposition depends only on the axis direction, so the result must match
# the unit-axis result and stay finite for every representable magnitude.
@pytest.mark.parametrize(
    "scale",
    [1e-160, 1e-200, 1e-300, 3.7e-308, 1e-320, 5e-324, 1e160, 1e300],
)
@pytest.mark.parametrize("axis", [pr.unitx, pr.unity, pr.unitz])
def test_subnormal_and_extreme_axis_scale(scale, axis):
    q = pr.random_quaternion(np.random.default_rng(53))
    swing_ref, twist_ref = pr.swing_twist_decomposition(q, axis)
    swing, twist = pr.swing_twist_decomposition(q, scale * axis)
    assert np.all(np.isfinite(swing)) and np.all(np.isfinite(twist))
    assert_array_almost_equal(swing, swing_ref)
    assert_array_almost_equal(twist, twist_ref)


def test_accepts_list_input():
    swing, twist = pr.swing_twist_decomposition(
        [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    )
    pr.assert_quaternion_equal(swing, pr.q_id)
    pr.assert_quaternion_equal(twist, pr.q_id)


def test_non_unit_quaternion_is_normalized():
    # check_quaternion normalizes, so a scaled quaternion gives the same
    # result as the normalized one.
    q = pr.random_quaternion(np.random.default_rng(21))
    axis = pr.unity
    swing_ref, twist_ref = pr.swing_twist_decomposition(q, axis)
    swing, twist = pr.swing_twist_decomposition(5.0 * q, axis)
    assert_array_almost_equal(swing, swing_ref)
    assert_array_almost_equal(twist, twist_ref)


# ---------------------------------------------------------------------------
# The eps parameter
# ---------------------------------------------------------------------------


def test_short_nonzero_axis_is_accepted():
    # Only an exactly-zero axis is rejected; a short but non-zero axis is
    # normalized like any other and does not depend on eps.
    q = pr.random_quaternion(np.random.default_rng(41))
    short_axis = np.array([1e-6, 0.0, 0.0])
    swing, twist = pr.swing_twist_decomposition(q, short_axis, eps=0.5)
    ref_swing, ref_twist = pr.swing_twist_decomposition(q, pr.unitx)
    assert_array_almost_equal(swing, ref_swing)
    assert_array_almost_equal(twist, ref_twist)


def test_eps_controls_singularity_fallback():
    # A rotation just short of a pi rotation orthogonal to the twist axis has
    # a small but non-zero twist norm (~1e-4 here).
    axis = pr.unitz
    q = pr.quaternion_from_axis_angle(np.hstack((pr.unitx, [np.pi - 2e-4])))

    # A large eps forces the singular fallback: the twist becomes the exact
    # identity quaternion.
    _, twist_fallback = pr.swing_twist_decomposition(q, axis, eps=1e-2)
    assert_array_almost_equal(twist_fallback, pr.q_id)

    # A tiny eps keeps the (normalized) twist and still reconstructs q.
    swing, twist = pr.swing_twist_decomposition(q, axis, eps=1e-12)
    assert np.linalg.norm(twist) == pytest.approx(1.0)
    assert pr.quaternion_dist(
        pr.concatenate_quaternions(swing, twist), q
    ) == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "axis", [np.zeros(3), [0.0, 0.0, 0.0], np.array([0.0, -0.0, 0.0])]
)
def test_zero_axis_raises(axis):
    with pytest.raises(ValueError, match="zero vector"):
        pr.swing_twist_decomposition(pr.q_id, axis)


@pytest.mark.parametrize("bad_q", [np.zeros(3), np.ones(5), np.ones((2, 4))])
def test_invalid_quaternion_shape_raises(bad_q):
    with pytest.raises(ValueError, match="quaternion"):
        pr.swing_twist_decomposition(bad_q, pr.unitz)
