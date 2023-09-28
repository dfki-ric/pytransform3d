"""Position+quaternion operations."""
import numpy as np
from .. import rotations
from ._utils import check_pq


def pq_slerp(start, end, t):
    """Spherical linear interpolation of position and quaternion.

    We will use spherical linear interpolation (SLERP) for the quaternion and
    linear interpolation for the position.

    An alternative approach is screw linear interpolation (ScLERP) with dual
    quaternions (see
    :func:`pytransform3d.transformations.dual_quaternion_sclerp`).

    Parameters
    ----------
    start : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    end : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)

    t : float in [0, 1]
        Position between start and end

    Returns
    -------
    pq_t : array-like, shape (7,)
        Position and orientation quaternion: (x, y, z, qw, qx, qy, qz)
    """
    start = check_pq(start)
    end = check_pq(end)
    start_p, start_q = np.array_split(start, (3,))
    end_p, end_q = np.array_split(end, (3,))
    q_t = rotations.quaternion_slerp(start_q, end_q, t, shortest_path=True)
    p_t = start_p + t * (end_p - start_p)
    return np.hstack((p_t, q_t))
