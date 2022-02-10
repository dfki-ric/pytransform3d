"""Plotting functions."""
import warnings
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from ._layout import make_3d_axis
from ._artists import Arrow3D
from ..transformations import transform, vectors_to_points
from ..rotations import unitx, unitz, perpendicular_to_vectors, norm_vector


def plot_box(ax=None, size=np.ones(3), A2B=np.eye(4), ax_s=1, wireframe=True,
             color="k", alpha=1.0):
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

    corners = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    corners = (corners - 0.5) * size
    corners = transform(
        A2B, np.hstack((corners, np.ones((len(corners), 1)))))[:, :3]

    if wireframe:
        for i, j in [(0, 1), (0, 2), (1, 3), (2, 3),
                     (4, 5), (4, 6), (5, 7), (6, 7),
                     (0, 4), (1, 5), (2, 6), (3, 7)]:
            ax.plot([corners[i, 0], corners[j, 0]],
                    [corners[i, 1], corners[j, 1]],
                    [corners[i, 2], corners[j, 2]],
                    c=color, alpha=alpha)
    else:
        p3c = Poly3DCollection(np.array([
            [corners[0], corners[1], corners[2]],
            [corners[1], corners[2], corners[3]],

            [corners[4], corners[5], corners[6]],
            [corners[5], corners[6], corners[7]],

            [corners[0], corners[1], corners[4]],
            [corners[1], corners[4], corners[5]],

            [corners[2], corners[6], corners[7]],
            [corners[2], corners[3], corners[7]],

            [corners[0], corners[4], corners[6]],
            [corners[0], corners[2], corners[6]],

            [corners[1], corners[5], corners[7]],
            [corners[1], corners[3], corners[7]],
        ]))
        p3c.set_alpha(alpha)
        p3c.set_facecolor(color)
        ax.add_collection3d(p3c)

    return ax


def plot_sphere(ax=None, radius=1.0, p=np.zeros(3), ax_s=1, wireframe=True,
                n_steps=20, alpha=1.0, color="k"):
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

    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = p[0] + radius * np.sin(phi) * np.cos(theta)
    y = p[1] + radius * np.sin(phi) * np.sin(theta)
    z = p[2] + radius * np.cos(phi)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=10, cstride=10, color=color, alpha=alpha)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_cylinder(ax=None, length=1.0, radius=1.0, thickness=0.0,
                  A2B=np.eye(4), ax_s=1, wireframe=True, n_steps=100,
                  alpha=1.0, color="k"):
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
        raise ValueError("Thickness of cylindrical shell results in "
                         "invalid inner radius: %g" % inner_radius)

    axis_start = A2B.dot(np.array([0, 0, -0.5 * length, 1]))[:3]
    axis_end = A2B.dot(np.array([0, 0, 0.5 * length, 1]))[:3]
    axis = axis_end - axis_start
    axis /= length

    not_axis = np.array([1, 0, 0])
    if (axis == not_axis).all():
        not_axis = np.array([0, 1, 0])

    n1 = np.cross(axis, not_axis)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis, n1)

    if wireframe:
        t = np.linspace(0, length, n_steps)
    else:
        t = np.array([0, length])
    theta = np.linspace(0, 2 * np.pi, n_steps)
    t, theta = np.meshgrid(t, theta)

    if thickness > 0.0:
        X_outer, Y_outer, Z_outer = [
            axis_start[i] + axis[i] * t
            + radius * np.sin(theta) * n1[i]
            + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        X_inner, Y_inner, Z_inner = [
            axis_end[i] - axis[i] * t
            + inner_radius * np.sin(theta) * n1[i]
            + inner_radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        X = np.hstack((X_outer, X_inner))
        Y = np.hstack((Y_outer, Y_inner))
        Z = np.hstack((Z_outer, Z_inner))
    else:
        X, Y, Z = [axis_start[i] + axis[i] * t
                   + radius * np.sin(theta) * n1[i]
                   + radius * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    if wireframe:
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, alpha=alpha,
                          color=color)
    else:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_mesh(ax=None, filename=None, A2B=np.eye(4),
              s=np.array([1.0, 1.0, 1.0]), ax_s=1, wireframe=False,
              convex_hull=False, alpha=1.0, color="k"):
    """Plot mesh.

    Note that this function requires the additional library 'trimesh'.
    It will print a warning if trimesh is not available.

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
            "package directory.")
        return ax

    try:
        import trimesh
    except ImportError:
        warnings.warn(
            "Cannot display mesh. Library 'trimesh' not installed.")
        return ax

    mesh = trimesh.load(filename)
    if convex_hull:
        mesh = mesh.convex_hull
    vertices = mesh.vertices * s
    vertices = np.hstack((vertices, np.ones((len(vertices), 1))))
    vertices = transform(A2B, vertices)[:, :3]
    vectors = np.array([vertices[[i, j, k]] for i, j, k in mesh.faces])
    if wireframe:
        surface = Line3DCollection(vectors)
        surface.set_color(color)
    else:
        surface = Poly3DCollection(vectors)
        surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)
    return ax


def plot_ellipsoid(ax=None, radii=np.ones(3), A2B=np.eye(4), ax_s=1,
                   wireframe=True, n_steps=20, alpha=1.0, color="k"):
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

    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = radius_x * np.sin(phi) * np.cos(theta)
    y = radius_y * np.sin(phi) * np.sin(theta)
    z = radius_z * np.cos(phi)

    shape = x.shape

    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    P = transform(A2B, vectors_to_points(P))[:, :3]

    x = P[:, 0].reshape(*shape)
    y = P[:, 1].reshape(*shape)
    z = P[:, 2].reshape(*shape)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=10, cstride=10, color=color, alpha=alpha)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_capsule(ax=None, A2B=np.eye(4), height=1.0, radius=1.0,
                 ax_s=1, wireframe=True, n_steps=20, alpha=1.0, color="k"):
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

    phi, theta = np.mgrid[0.0:np.pi:n_steps * 1j, 0.0:2.0 * np.pi:n_steps * 1j]
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    z[len(z) // 2:] -= 0.5 * height
    z[:len(z) // 2] += 0.5 * height

    shape = x.shape

    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    P = transform(A2B, vectors_to_points(P))[:, :3]

    x = P[:, 0].reshape(*shape)
    y = P[:, 1].reshape(*shape)
    z = P[:, 2].reshape(*shape)

    if wireframe:
        ax.plot_wireframe(
            x, y, z, rstride=10, cstride=10, color=color, alpha=alpha)
    else:
        ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_cone(ax=None, height=1.0, radius=1.0, A2B=np.eye(4), ax_s=1,
              wireframe=True, n_steps=20, alpha=1.0, color="k"):
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

    axis_start = A2B.dot(np.array([0, 0, 0, 1]))[:3]
    axis_end = A2B.dot(np.array([0, 0, height, 1]))[:3]
    axis = axis_end - axis_start
    axis /= height

    not_axis = np.array([1, 0, 0])
    if (axis == not_axis).all():
        not_axis = np.array([0, 1, 0])

    n1 = np.cross(axis, not_axis)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(axis, n1)

    if wireframe:
        t = np.linspace(0, height, n_steps)
        radii = np.linspace(radius, 0, n_steps)
    else:
        t = np.array([0, height])
        radii = np.array([radius, 0])
    theta = np.linspace(0, 2 * np.pi, n_steps)
    t, theta = np.meshgrid(t, theta)

    X, Y, Z = [axis_start[i] + axis[i] * t
               + radii * np.sin(theta) * n1[i]
               + radii * np.cos(theta) * n2[i] for i in [0, 1, 2]]

    if wireframe:
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, alpha=alpha,
                          color=color)
    else:
        ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

    return ax


def plot_vector(ax=None, start=np.zeros(3), direction=np.array([1, 0, 0]),
                s=1.0, arrowstyle="simple", ax_s=1, **kwargs):
    """Plot Vector.

    Draws an arrow from start to start + s * direction.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    start : array-like, shape (3,), optional (default: [0, 0, 0])
        Start of the vector

    direction : array-like, shape (3,), optional (default: [1, 0, 0])
        Direction of the vector

    s : float, optional (default: 1)
        Scaling of the vector that will be drawn

    arrowstyle : str, or ArrowStyle, optional (default: 'simple')
        See matplotlib's documentation of arrowstyle in
        matplotlib.patches.FancyArrowPatch for more options

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    axis_arrow = Arrow3D(
        [start[0], start[0] + s * direction[0]],
        [start[1], start[1] + s * direction[1]],
        [start[2], start[2] + s * direction[2]],
        mutation_scale=20, arrowstyle=arrowstyle, **kwargs)
    ax.add_artist(axis_arrow)

    return ax


def plot_length_variable(ax=None, start=np.zeros(3), end=np.ones(3), name="l",
                         above=False, ax_s=1, color="k", **kwargs):
    """Plot length with text at its center.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created

    start : array-like, shape (3,), optional (default: [0, 0, 0])
        Start point

    end : array-like, shape (3,), optional (default: [1, 1, 1])
        End point

    name : str, optional (default: 'l')
        Text in the middle

    above : bool, optional (default: False)
        Plot name above line

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    color : str, optional (default: black)
        Color in which the cylinder should be plotted

    kwargs : dict, optional (default: {})
        Additional arguments for the text, e.g. fontsize

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis
    """
    if ax is None:
        ax = make_3d_axis(ax_s)

    direction = end - start
    length = np.linalg.norm(direction)

    if above:
        ax.plot([start[0], end[0]], [start[1], end[1]],
                [start[2], end[2]], color=color)
    else:
        mid1 = start + 0.4 * direction
        mid2 = start + 0.6 * direction
        ax.plot([start[0], mid1[0]], [start[1], mid1[1]],
                [start[2], mid1[2]], color=color)
        ax.plot([end[0], mid2[0]], [end[1], mid2[1]],
                [end[2], mid2[2]], color=color)

    if np.linalg.norm(direction / length - unitz) < np.finfo(float).eps:
        axis = unitx
    else:
        axis = unitz

    mark = (norm_vector(perpendicular_to_vectors(direction, axis))
            * 0.03 * length)
    mark_start1 = start + mark
    mark_start2 = start - mark
    mark_end1 = end + mark
    mark_end2 = end - mark
    ax.plot([mark_start1[0], mark_start2[0]],
            [mark_start1[1], mark_start2[1]],
            [mark_start1[2], mark_start2[2]],
            color=color)
    ax.plot([mark_end1[0], mark_end2[0]],
            [mark_end1[1], mark_end2[1]],
            [mark_end1[2], mark_end2[2]],
            color=color)
    text_location = start + 0.45 * direction
    if above:
        text_location[2] += 0.3 * length
    ax.text(text_location[0], text_location[1], text_location[2], name,
            zdir="x", **kwargs)

    return ax
