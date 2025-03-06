import numpy as np

from .._geometry import unit_sphere_surface_grid
from ..trajectories import (
    transforms_from_exponential_coordinates,
)
from ..transformations import (
    transform_from,
)


def to_ellipsoid(mean, cov):
    """Compute error ellipsoid.

    An error ellipsoid indicates the equiprobable surface. The resulting
    ellipsoid includes one standard deviation of the data along each main
    axis, which covers approximately 68.27% of the data. Multiplying the
    radii with factors > 1 will increase the coverage. The usual factors
    for Gaussian distributions apply:

    * 1 - 68.27%
    * 1.65 - 90%
    * 1.96 - 95%
    * 2 - 95.45%
    * 2.58 - 99%
    * 3 - 99.73%

    Parameters
    ----------
    mean : array-like, shape (3,)
        Mean of distribution.

    cov : array-like, shape (3, 3)
        Covariance of distribution.

    Returns
    -------
    ellipsoid2origin : array, shape (4, 4)
        Ellipsoid frame in world frame. Note that there are multiple solutions
        possible for the orientation because an ellipsoid is symmetric.
        A body-fixed rotation around a main axis by 180 degree results in the
        same ellipsoid.

    radii : array, shape (3,)
        Radii of ellipsoid, coinciding with standard deviations along the
        three axes of the ellipsoid. These are sorted in ascending order.
    """
    from scipy import linalg

    radii, R = linalg.eigh(cov)
    if np.linalg.det(R) < 0:  # undo reflection (exploit symmetry)
        R *= -1
    ellipsoid2origin = transform_from(R=R, p=mean)
    return ellipsoid2origin, np.sqrt(np.abs(radii))


def to_projected_ellipsoid(mean, cov, factor=1.96, n_steps=20):
    """Compute projected error ellipsoid.

    An error ellipsoid shows equiprobable points. This is a projection of a
    Gaussian distribution in exponential coordinate space to 3D.

    Parameters
    ----------
    mean : array-like, shape (4, 4)
        Mean of pose distribution.

    cov : array-like, shape (6, 6)
        Covariance of pose distribution in exponential coordinate space.

    factor : float, optional (default: 1.96)
        Multiple of the standard deviations that should be plotted.

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension.

    Returns
    -------
    x : array, shape (n_steps, n_steps)
        Coordinates on x-axis of grid on projected ellipsoid.

    y : array, shape (n_steps, n_steps)
        Coordinates on y-axis of grid on projected ellipsoid.

    z : array, shape (n_steps, n_steps)
        Coordinates on z-axis of grid on projected ellipsoid.
    """
    from scipy import linalg

    vals, vecs = linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    radii = factor * np.sqrt(vals[:3])

    # Grid on ellipsoid in exponential coordinate space
    x, y, z = unit_sphere_surface_grid(n_steps)
    x *= radii[0]
    y *= radii[1]
    z *= radii[2]
    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    P = np.dot(P, vecs[:, :3].T)

    # Grid in Cartesian space
    T_diff = transforms_from_exponential_coordinates(P)
    # same as T_diff[m, :3, :3].T.dot(T_diff[m, :3, 3]) for each m
    P = np.einsum("ikj,ik->ij", T_diff[:, :3, :3], T_diff[:, :3, 3])
    P = (np.dot(P, mean[:3, :3].T) + mean[np.newaxis, :3, 3]).T

    shape = x.shape
    x = P[0].reshape(*shape)
    y = P[1].reshape(*shape)
    z = P[2].reshape(*shape)

    return x, y, z


def plot_projected_ellipsoid(
    ax,
    mean,
    cov,
    factor=1.96,
    wireframe=True,
    n_steps=20,
    color=None,
    alpha=1.0,
):  # pragma: no cover
    """Plots projected equiprobable ellipsoid in 3D.

    An error ellipsoid shows equiprobable points. This is a projection of a
    Gaussian distribution in exponential coordinate space to 3D.

    Parameters
    ----------
    ax : axis
        Matplotlib axis.

    mean : array-like, shape (4, 4)
        Mean pose.

    cov : array-like, shape (6, 6)
        Covariance in exponential coordinate space.

    factor : float, optional (default: 1.96)
        Multiple of the standard deviations that should be plotted.

    wireframe : bool, optional (default: True)
        Plot wireframe of ellipsoid and surface otherwise.

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension.

    color : str, optional (default: None)
        Color in which the equiprobably lines should be plotted.

    alpha : float, optional (default: 1.0)
        Alpha value for lines.

    Returns
    -------
    ax : axis
        Matplotlib axis.
    """
    x, y, z = to_projected_ellipsoid(mean, cov, factor, n_steps)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=2, cstride=2, color=color, alpha=alpha
        )
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax
