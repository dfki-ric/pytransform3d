"""Modified Rodrigues parameters."""
import numpy as np
from ._utils import check_mrp


def concatenate_mrp(mrp1, mrp2):
    r"""Concatenate two rotations defined by modified Rodrigues parameters.

    Suppose we want to apply two extrinsic rotations given by modified
    Rodrigues parameters mrp1 and mrp2 to a vector v. We can either apply mrp2
    to v and then mrp1 to the result or we can concatenate mrp1 and mrp2 and
    apply the result to v.

    The solution for concatenation of two rotations
    :math:`\boldsymbol{p}_1,\boldsymbol{p}_2` is given by Shuster [1]_:

    .. math::

        \boldsymbol{p} =
        \frac{
        (1 - ||\boldsymbol{p}_1||^2) \boldsymbol{p}_2
        + (1 - ||\boldsymbol{p}_2||^2) \boldsymbol{p}_1
        - 2 \boldsymbol{p}_2 \times \boldsymbol{p}_1}
        {1 + ||\boldsymbol{p}_2||^2 ||\boldsymbol{p}_1||^2
        - 2 \boldsymbol{p}_2 \cdot \boldsymbol{p}_1}.

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
