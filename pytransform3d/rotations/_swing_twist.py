"""Swing-twist decomposition of a rotation."""

import numpy as np

from ._quaternion import check_quaternion, concatenate_quaternions, q_conj
from ._utils import norm_vector


def swing_twist_decomposition(q, axis, eps=np.finfo(float).eps):
    r"""Decompose quaternion into swing and twist about an axis.

    Any rotation :math:`\boldsymbol{q}` can be split into a *twist* about a
    given axis :math:`\boldsymbol{e}` and a *swing* that rotates the axis
    itself, such that

    .. math::

        \boldsymbol{q} =
        \boldsymbol{q}_\text{swing} \boldsymbol{q}_\text{twist},

    where the twist is applied first. The twist
    :math:`\boldsymbol{q}_\text{twist}` is a rotation about
    :math:`\boldsymbol{e}` and the swing :math:`\boldsymbol{q}_\text{swing}`
    is a rotation about an axis that is orthogonal to :math:`\boldsymbol{e}`
    [1]_.

    The twist is obtained by projecting the vector part of the quaternion
    onto the twist axis and renormalizing; the swing is then recovered as
    :math:`\boldsymbol{q}_\text{swing} = \boldsymbol{q}
    \boldsymbol{q}_\text{twist}^{-1}`.

    This decomposition is useful, for example, to enforce joint limits, to
    separate the roll about a link axis from the remaining orientation, or to
    isolate the rotation about a symmetry axis.

    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z).

    axis : array-like, shape (3,)
        Twist axis. Does not have to be normalized.

    eps : float, optional (default: np.finfo(float).eps)
        Threshold for degeneracy detection


    Returns
    -------
    swing : array, shape (4,)
        Unit quaternion that represents the swing, a rotation about an axis
        orthogonal to the twist axis.

    twist : array, shape (4,)
        Unit quaternion that represents the twist, a rotation about the given
        axis.

    Raises
    ------
    ValueError
        If the twist axis is the zero vector.

    See Also
    --------
    swing_twist_composition
        Reconstructs the original rotation from swing and twist.

    concatenate_quaternions
        Reconstructs the original rotation from swing and twist via
        ``concatenate_quaternions(swing, twist)``.

    Notes
    -----
    Beyond the algorithm of [1]_, this implementation fixes two conventions
    for cases the paper leaves open:

    1. **Degenerate twist.** When ``twist_norm < eps``, both the scalar part
       and the projection of the vector part onto the twist axis vanish. This
       happens for a rotation by :math:`\pi` about an axis orthogonal to the
       twist axis, where the twist is undefined. We resolve it by returning the
       identity as the twist and letting the swing carry the full rotation.

    2. **Canonical hemisphere.** The twist is normalized to a non-negative
       scalar part (:math:`w \geq 0`). Since :math:`\boldsymbol{q}` and
       :math:`-\boldsymbol{q}` represent the same rotation, this makes the
       returned twist unique and keeps the twist angle in
       :math:`[-\pi, \pi]`.

    References
    ----------
    .. [1] Dobrowolski, P. (2015). Swing-twist decomposition in Clifford
       algebra. https://arxiv.org/abs/1506.05481
    """

    q = check_quaternion(q, unit=True)

    axis = np.asarray(axis, dtype=float)

    # Rescale by the largest-magnitude component before normalizing. Squaring
    # the raw entries in np.linalg.norm underflows to zero for subnormal axes
    # (spuriously raising below) or loses precision in the subnormal range
    # (yielding a non-unit axis). Dividing by the max component first brings
    # the entries to order 1, so the subsequent normalization stays exact and
    # scale-invariant down to the smallest representable magnitude.

    scale = np.max(np.abs(axis))
    if scale == 0.0:
        raise ValueError("Twist axis must not be the zero vector")

    axis = norm_vector(axis / scale)

    projection = np.dot(q[1:], axis) * axis
    twist = np.array([q[0], projection[0], projection[1], projection[2]])
    twist_norm = np.linalg.norm(twist)

    if twist_norm < eps:
        # Degenerate case: both the scalar part and the projection onto the
        # axis vanish, which happens for a rotation by pi about an axis
        # orthogonal to the twist axis. The twist is then undefined, so we
        # return the identity and let the swing carry the full rotation.
        twist = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        twist /= twist_norm
        if twist[0] < 0.0:
            # Flip to the canonical hemisphere (w >= 0) so the twist angle
            # stays in [-pi, pi].
            twist = -twist

    swing = concatenate_quaternions(q, q_conj(twist))
    return swing, twist


def swing_twist_composition(swing, twist):
    r"""Recompose a rotation from its swing and twist components.

    This is the inverse of :func:`swing_twist_decomposition`. Given a swing
    and a twist quaternion, the original rotation is recovered by

    .. math::

        \boldsymbol{q} = \boldsymbol{q}_{swing} \boldsymbol{q}_{twist},

    i.e. the twist is applied first and then the swing.

    Note that the reconstructed quaternion may differ in sign from the
    quaternion that was originally decomposed, since :math:`\boldsymbol{q}`
    and :math:`-\boldsymbol{q}` represent the same rotation.

    Parameters
    ----------
    swing : array-like, shape (4,)
        Unit quaternion that represents the swing, a rotation about an axis
        orthogonal to the twist axis.

    twist : array-like, shape (4,)
        Unit quaternion that represents the twist, a rotation about the twist
        axis.

    Returns
    -------
    q : array, shape (4,)
        Unit quaternion that represents the recomposed rotation: (w, x, y, z).

    See Also
    --------
    swing_twist_decomposition
        Decomposes a rotation into swing and twist.
    """
    swing = check_quaternion(swing, unit=True)
    twist = check_quaternion(twist, unit=True)
    return concatenate_quaternions(swing, twist)
