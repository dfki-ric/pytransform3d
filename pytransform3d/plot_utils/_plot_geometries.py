"""Plotting functions for geometries."""

import warnings

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from ._layout import make_3d_axis
from .._geometry import unit_sphere_surface_grid, transform_surface
from .._mesh_loader import load_mesh
from ..transformations import transform


def plot_box(
    ax=None,
    size=np.ones(3),
    A2B=np.eye(4),
    ax_s=1,
    wireframe=True,
    color="k",
    alpha=1.0,
):
    """Plot box.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    size : array-like, shape (3,), optional (default: [1, 1, 1])
        Size of the box per dimension

    A2B : array-like, shape (4, 4)
        Center of the box

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of box and surface otherwise

    color : str, optional (default: black)
        Color in which the box should be plotted

    alpha : float, optional (default: 1)
        Alpha value of the mesh that will be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    vertices = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
    vertices = (np.array(vertices, dtype=float) - 0.5) * size
    vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    vertices = transform(A2B, vertices)[:, :3]

    if wireframe:
        connections = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        surface = Line3DCollection(vertices[connections])
        surface.set_color(color)
    else:
        faces = [
            [0, 1, 2],
            [1, 2, 3],
            [4, 5, 6],
            [5, 6, 7],
            [0, 1, 4],
            [1, 4, 5],
            [2, 6, 7],
            [2, 3, 7],
            [0, 4, 6],
            [0, 2, 6],
            [1, 5, 7],
            [1, 3, 7],
        ]
        surface = Poly3DCollection(vertices[faces])
        surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)

    return ax


def plot_sphere(
    ax=None,
    radius=1.0,
    p=np.zeros(3),
    ax_s=1,
    wireframe=True,
    n_steps=20,
    alpha=1.0,
    color="k",
):
    """Plot sphere.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    radius : float, optional (default: 1)
        Radius of the sphere

    p : array-like, shape (3,), optional (default: [0, 0, 0])
        Center of the sphere

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of sphere and surface otherwise

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension

    alpha : float, optional (default: 1)
        Alpha value of the sphere that will be plotted

    color : str, optional (default: black)
        Color in which the sphere should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    x, y, z = unit_sphere_surface_grid(n_steps)
    x = p[0] + radius * x
    y = p[1] + radius * y
    z = p[2] + radius * z

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=2, cstride=2, color=color, alpha=alpha
        )
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_spheres(
    ax=None,
    radius=np.ones(1),
    p=np.zeros((1, 3)),
    ax_s=1,
    wireframe=True,
    n_steps=20,
    alpha=np.ones(1),
    color=np.zeros((1, 3)),
):
    """Plot multiple spheres.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    radius : array-like, shape (n_spheres,), optional (default: 1)
        Radius of the sphere(s)

    p : array-like, shape (n_spheres, 3), optional (default: [0, 0, 0])
        Center of the sphere(s)

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of sphere(s) and surface otherwise

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension

    alpha : array-like, shape (n_spheres,), optional (default: 1)
        Alpha value of the sphere(s) that will be plotted

    color : array-like, shape (n_spheres, 3), optional (default: black)
        Color in which the sphere(s) should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    radius = np.asarray(radius)
    p = np.asarray(p)

    phi, theta = np.mgrid[
        0.0 : np.pi : n_steps * 1j, 0.0 : 2.0 * np.pi : n_steps * 1j
    ]
    sin_phi = np.sin(phi)
    verts = (
        radius[..., np.newaxis, np.newaxis, np.newaxis]
        * np.array(
            [sin_phi * np.cos(theta), sin_phi * np.sin(theta), np.cos(phi)]
        )[np.newaxis, ...]
        + p[..., np.newaxis, np.newaxis]
    )
    colors = np.resize(color, (len(verts), 3))
    alphas = np.resize(alpha, len(verts))

    for verts_i, color_i, alpha_i in zip(verts, colors, alphas):
        if wireframe:
            ax.plot_wireframe(
                *verts_i, rstride=2, cstride=2, color=color_i, alpha=alpha_i
            )
        else:
            ax.plot_surface(*verts_i, color=color_i, alpha=alpha_i, linewidth=0)

    return ax


def plot_cylinder(
    ax=None,
    length=1.0,
    radius=1.0,
    thickness=0.0,
    A2B=np.eye(4),
    ax_s=1,
    wireframe=True,
    n_steps=100,
    alpha=1.0,
    color="k",
):
    """Plot cylinder.

    A cylinder is the volume covered by a disk moving along a line segment.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    length : float, optional (default: 1)
        Length of the cylinder

    radius : float, optional (default: 1)
        Radius of the cylinder

    thickness : float, optional (default: 0)
        Thickness of a cylindrical shell. It will be subtracted from the
        outer radius to obtain the inner radius. The difference must be
        greater than 0.

    A2B : array-like, shape (4, 4)
        Center of the cylinder

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of cylinder and surface otherwise

    n_steps : int, optional (default: 100)
        Number of discrete steps plotted in each dimension

    alpha : float, optional (default: 1)
        Alpha value of the cylinder that will be plotted

    color : str, optional (default: black)
        Color in which the cylinder should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis

    Raises
    ------
    ValueError
        If thickness is <= 0
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    inner_radius = radius - thickness
    if inner_radius <= 0.0:
        raise ValueError(
            "Thickness of cylindrical shell results in "
            "invalid inner radius: %g" % inner_radius
        )

    if wireframe:
        t = np.linspace(0, length, n_steps)
    else:
        t = np.array([0, length])
    angles = np.linspace(0, 2 * np.pi, n_steps)
    t, angles = np.meshgrid(t, angles)

    A2B = np.asarray(A2B)
    axis_start = np.dot(A2B, [0, 0, -0.5 * length, 1])[:3]
    X, Y, Z = _elongated_circular_grid(axis_start, A2B, t, radius, angles)
    if thickness > 0.0:
        A2B_left_hand = np.copy(A2B)
        A2B_left_hand[:3, 2] *= -1.0
        axis_end = np.dot(A2B, [0, 0, 0.5 * length, 1])[:3]
        X_inner, Y_inner, Z_inner = _elongated_circular_grid(
            axis_end, A2B_left_hand, t, inner_radius, angles
        )
        X = np.hstack((X, X_inner))
        Y = np.hstack((Y, Y_inner))
        Z = np.hstack((Z, Z_inner))

    if wireframe:
        ax.plot_wireframe(
            X, Y, Z, rstride=10, cstride=10, alpha=alpha, color=color
        )
    else:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_mesh(
    ax=None,
    filename=None,
    A2B=np.eye(4),
    s=np.array([1.0, 1.0, 1.0]),
    ax_s=1,
    wireframe=False,
    convex_hull=False,
    alpha=1.0,
    color="k",
):
    """Plot mesh.

    Note that this function requires the additional library to load meshes
    such as trimesh or open3d.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    filename : str, optional (default: None)
        Path to mesh file.

    A2B : array-like, shape (4, 4)
        Pose of the mesh

    s : array-like, shape (3,), optional (default: [1, 1, 1])
        Scaling of the mesh that will be drawn

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of mesh and surface otherwise

    convex_hull : bool, optional (default: False)
        Show convex hull instead of the original mesh. This can be much
        faster.

    alpha : float, optional (default: 1)
        Alpha value of the mesh that will be plotted

    color : str, optional (default: black)
        Color in which the mesh should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    if filename is None:
        warnings.warn(
            "No filename given for mesh. When you use the "
            "UrdfTransformManager, make sure to set the mesh path or "
            "package directory.",
            UserWarning,
            stacklevel=2,
        )
        return ax

    mesh = load_mesh(filename)
    if convex_hull:
        mesh.convex_hull()

    vertices = np.asarray(mesh.vertices) * s
    vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    vertices = transform(A2B, vertices)[:, :3]
    faces = vertices[mesh.triangles]
    if wireframe:
        surface = Line3DCollection(faces)
        surface.set_color(color)
    else:
        surface = Poly3DCollection(faces)
        surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)
    return ax


def plot_ellipsoid(
    ax=None,
    radii=np.ones(3),
    A2B=np.eye(4),
    ax_s=1,
    wireframe=True,
    n_steps=20,
    alpha=1.0,
    color="k",
):
    """Plot ellipsoid.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    radii : array-like, shape (3,)
        Radii along the x-axis, y-axis, and z-axis of the ellipsoid.

    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of ellipsoid and surface otherwise

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension

    alpha : float, optional (default: 1)
        Alpha value of the ellipsoid that will be plotted

    color : str, optional (default: black)
        Color in which the ellipsoid should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    radius_x, radius_y, radius_z = radii

    x, y, z = unit_sphere_surface_grid(n_steps)
    x *= radius_x
    y *= radius_y
    z *= radius_z

    x, y, z = transform_surface(A2B, x, y, z)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=2, cstride=2, color=color, alpha=alpha
        )
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_capsule(
    ax=None,
    A2B=np.eye(4),
    height=1.0,
    radius=1.0,
    ax_s=1,
    wireframe=True,
    n_steps=20,
    alpha=1.0,
    color="k",
):
    """Plot capsule.

    A capsule is the volume covered by a sphere moving along a line segment.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    A2B : array-like, shape (4, 4)
        Frame of the capsule, located at the center of the line segment.

    height : float, optional (default: 1)
        Height of the capsule along its z-axis.

    radius : float, optional (default: 1)
        Radius of the capsule.

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of capsule and surface otherwise

    n_steps : int, optional (default: 20)
        Number of discrete steps plotted in each dimension

    alpha : float, optional (default: 1)
        Alpha value of the mesh that will be plotted

    color : str, optional (default: black)
        Color in which the capsule should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    x, y, z = unit_sphere_surface_grid(n_steps)
    x *= radius
    y *= radius
    z *= radius
    z[len(z) // 2 :] -= 0.5 * height
    z[: len(z) // 2] += 0.5 * height

    x, y, z = transform_surface(A2B, x, y, z)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=2, cstride=2, color=color, alpha=alpha
        )
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_cone(
    ax=None,
    height=1.0,
    radius=1.0,
    A2B=np.eye(4),
    ax_s=1,
    wireframe=True,
    n_steps=20,
    alpha=1.0,
    color="k",
):
    """Plot cone.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    height : float, optional (default: 1)
        Height of the cone

    radius : float, optional (default: 1)
        Radius of the cone

    A2B : array-like, shape (4, 4)
        Center of the cone

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    wireframe : bool, optional (default: True)
        Plot wireframe of cone and surface otherwise

    n_steps : int, optional (default: 100)
        Number of discrete steps plotted in each dimension

    alpha : float, optional (default: 1)
        Alpha value of the cone that will be plotted

    color : str, optional (default: black)
        Color in which the cone should be plotted

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    if wireframe:
        t = np.linspace(0, height, n_steps)
        radii = np.linspace(radius, 0, n_steps)
    else:
        t = np.array([0, height])
        radii = np.array([radius, 0])
    angles = np.linspace(0, 2 * np.pi, n_steps)
    t, angles = np.meshgrid(t, angles)

    A2B = np.asarray(A2B)
    X, Y, Z = _elongated_circular_grid(A2B[:3, 3], A2B, t, radii, angles)

    if wireframe:
        ax.plot_wireframe(
            X, Y, Z, rstride=5, cstride=5, alpha=alpha, color=color
        )
    else:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

    return ax


def _elongated_circular_grid(
    bottom_point, A2B, height_fractions, radii, angles
):
    return [
        bottom_point[i]
        + radii * np.sin(angles) * A2B[i, 0]
        + radii * np.cos(angles) * A2B[i, 1]
        + A2B[i, 2] * height_fractions
        for i in range(3)
    ]
