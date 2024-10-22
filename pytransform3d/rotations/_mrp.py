"""Modified Rodrigues parameters."""
import numpy as np
from ._utils import check_mrp, norm_angle
from ._conversions import axis_angle_from_mrp, mrp_from_axis_angle
from ._constants import two_pi


def norm_mrp(mrp):
    """Normalize angle of modified Rodrigues parameters to range [-pi, pi].

    Normalization of modified Rodrigues parameters is required to avoid the
    singularity at a rotation angle of 2 * pi.

    Parameters
    ----------
    mrp : array-like, shape (3,)
        Modified Rodrigues parameters.

    Returns
    -------
    mrp : array, shape (3,)
        Modified Rodrigues parameters with angle normalized to [-pi, pi].
    """
    mrp = check_mrp(mrp)
    a = axis_angle_from_mrp(mrp)
    a[3] = norm_angle(a[3])
    return mrp_from_axis_angle(a)


def mrp_near_singularity(mrp, tolerance=1e-6):
    """Check if modified Rodrigues parameters are close to singularity.

    MRPs have a singularity at 2 * pi, i.e., the norm approaches infinity as
    the angle approaches 2 * pi.

    Parameters
    ----------
    mrp : array-like, shape (3,)
        Modified Rodrigues parameters.

    tolerance : float, optional (default: 1e-6)
        Tolerance for check.

    Returns
    -------
    near_singularity : bool
        MRPs are near singularity.
    """
    check_mrp(mrp)
    mrp_norm = np.linalg.norm(mrp)
    angle = np.arctan(mrp_norm) * 4.0
    return abs(angle - two_pi) < tolerance


def mrp_double(mrp):
    r"""Other modified Rodrigues parameters representing the same orientation.

    MRPs have two representations for the same rotation:
    :math:`\boldsymbol{\psi}` and :math:`-\frac{1}{||\boldsymbol{\psi}||^2}
    \boldsymbol{\psi}` represent the same rotation and correspond to two
    antipodal unit quaternions [1]_.

    Parameters
    ----------
    mrp : array-like, shape (3,)
        Modified Rodrigues parameters.

    Returns
    -------
    mrp_double : array, shape (3,)
        Different modified Rodrigues parameters that represent the same
        orientation.

    References
    ----------
    .. [1] Shuster, M. D. (1993). A Survey of Attitude Representations.
       Journal of the Astronautical Sciences, 41, 439-517.
       http://malcolmdshuster.com/Pub_1993h_J_Repsurv_scan.pdf
    """
    mrp = check_mrp(mrp)
    return mrp / -np.dot(mrp, mrp)


def concatenate_mrp(mrp1, mrp2):
    r"""Concatenate two rotations defined by modified Rodrigues parameters.

    Suppose we want to apply two extrinsic rotations given by modified
    Rodrigues parameters mrp1 and mrp2 to a vector v. We can either apply mrp2
    to v and then mrp1 to the result or we can concatenate mrp1 and mrp2 and
    apply the result to v.

    The solution for concatenation of two rotations
    :math:`\boldsymbol{\psi}_1,\boldsymbol{\psi}_2` is given by Shuster [1]_:

    .. math::

        \boldsymbol{\psi} =
        \frac{
        (1 - ||\boldsymbol{\psi}_1||^2) \boldsymbol{\psi}_2
        + (1 - ||\boldsymbol{\psi}_2||^2) \boldsymbol{\psi}_1
        - 2 \boldsymbol{\psi}_2 \times \boldsymbol{\psi}_1}
        {1 + ||\boldsymbol{\psi}_2||^2 ||\boldsymbol{\psi}_1||^2
        - 2 \boldsymbol{\psi}_2 \cdot \boldsymbol{\psi}_1}.

    Parameters
    ----------
    mrp1 : array-like, shape (3,)
        Modified Rodrigues parameters.

    mrp2 : array-like, shape (3,)
        Modified Rodrigues parameters.

    Returns
    -------
    mrp12 : array, shape (3,)
        Modified Rodrigues parameters that represent the concatenated rotation
        of mrp1 and mrp2.

    References
    ----------
    .. [1] Shuster, M. D. (1993). A Survey of Attitude Representations.
       Journal of the Astronautical Sciences, 41, 439-517.
       http://malcolmdshuster.com/Pub_1993h_J_Repsurv_scan.pdf
    """
    mrp1 = check_mrp(mrp1)
    mrp2 = check_mrp(mrp2)
    norm1_sq = np.linalg.norm(mrp1) ** 2
    norm2_sq = np.linalg.norm(mrp2) ** 2
    cross_product = np.cross(mrp2, mrp1)
    scalar_product = np.dot(mrp2, mrp1)
    return (
        (1.0 - norm1_sq) * mrp2 + (1.0 - norm2_sq) * mrp1 - 2.0 * cross_product
    ) / (1.0 + norm2_sq * norm1_sq - 2.0 * scalar_product)
