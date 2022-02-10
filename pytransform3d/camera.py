"""Transformations related to cameras.

See :doc:`camera` for more information.
"""
import numpy as np
from .transformations import (invert_transform, transform, check_transform,
                              vectors_to_points)


def make_world_grid(n_lines=11, n_points_per_line=51, xlim=(-0.5, 0.5),
                    ylim=(-0.5, 0.5)):
    """Generate grid in world coordinate frame.

    The grid will have the form

    .. code::

        +----+----+----+----+----+
        |    |    |    |    |    |
        +----+----+----+----+----+
        |    |    |    |    |    |
        +----+----+----+----+----+
        |    |    |    |    |    |
        +----+----+----+----+----+
        |    |    |    |    |    |
        +----+----+----+----+----+
        |    |    |    |    |    |
        +----+----+----+----+----+

    on the x-y plane with z=0 for all points.

    Parameters
    ----------
    n_lines : int, optional (default: 11)
        Number of lines

    n_points_per_line : int, optional (default: 51)
        Number of points per line

    xlim : tuple, optional (default: (-0.5, 0.5))
        Range on x-axis

    ylim : tuple, optional (default: (-0.5, 0.5))
        Range on y-axis

    Returns
    -------
    world_grid : array-like, shape (2 * n_lines * n_points_per_line, 4)
        Grid as homogenous coordinate vectors
    """
    world_grid_x = np.vstack([make_world_line([xlim[0], y], [xlim[1], y],
                                              n_points_per_line)
                              for y in np.linspace(ylim[0], ylim[1], n_lines)])
    world_grid_y = np.vstack([make_world_line([x, ylim[0]], [x, ylim[1]],
                                              n_points_per_line)
                              for x in np.linspace(xlim[0], xlim[1], n_lines)])
    return np.vstack((world_grid_x, world_grid_y))


def make_world_line(p1, p2, n_points):
    """Generate line in world coordinate frame.

    Parameters
    ----------
    p1 : array-like, shape (2 or 3,)
        Start point of the line

    p2 : array-like, shape (2 or 3,)
        End point of the line

    n_points : int
        Number of points

    Returns
    -------
    line : array-like, shape (n_points, 4)
        Samples from line in world frame
    """
    if len(p1) == 2:
        p1 = [p1[0], p1[1], 0]
    if len(p2) == 2:
        p2 = [p2[0], p2[1], 0]
    return np.array([np.linspace(p1[0], p2[0], n_points),
                     np.linspace(p1[1], p2[1], n_points),
                     np.linspace(p1[2], p2[2], n_points),
                     np.ones(n_points)]).T


def cam2sensor(P_cam, focal_length, kappa=0.0):
    """Project points from 3D camera coordinate system to sensor plane.

    Parameters
    ----------
    P_cam : array-like, shape (n_points, 3 or 4)
        Points in camera coordinates

    focal_length : float
        Focal length of the camera

    kappa : float, optional (default: 0)
        Camera distortion parameter

    Returns
    -------
    P_sensor : array-like, shape (n_points, 2)
        Points on the sensor plane. The result for points that are behind the
        camera will be a vector of nans.

    Raises
    ------
    ValueError
        If input is not valid
    """
    n_points, n_dims = P_cam.shape
    if n_dims not in (3, 4):
        raise ValueError("Expected 3- or 4-dimensional points, got %d "
                         "dimensions" % n_dims)
    if focal_length <= 0.0:
        raise ValueError("Focal length must be greater than 0.")

    P_sensor = np.empty((n_points, 2))
    ahead = P_cam[:, 2] > 0.0
    P_sensor[ahead] = P_cam[ahead][:, :2] / P_cam[ahead][:, 2, np.newaxis]
    behind = np.logical_not(ahead)

    for n in range(P_sensor.shape[0]):
        P_sensor[n] *= 1.0 / (1.0 + kappa * np.linalg.norm(P_sensor[n]) ** 2)
    P_sensor *= focal_length
    P_sensor[behind] = np.nan
    return P_sensor


def sensor2img(P_sensor, sensor_size, image_size, image_center=None):
    """Project points from 2D sensor plane to image coordinate system.

    Parameters
    ----------
    P_sensor : array-like, shape (n_points, 2)
        Points on camera sensor

    sensor_size : array-like, shape (2,)
        Size of the sensor array

    image_size : array-like, shape (2,)
        Size of the camera image: (width, height)

    image_center : array-like, shape (2,), optional (default: image_size / 2)
        Center of the image

    Returns
    -------
    P_img : array-like, shape (n_points, 2)
        Points on image
    """
    P_img = np.asarray(image_size) * P_sensor / np.asarray(sensor_size)
    if image_center is None:
        image_center = np.asarray(image_size) / 2
    P_img += np.asarray(image_center)
    return P_img


def world2image(P_world, cam2world, sensor_size, image_size, focal_length,
                image_center=None, kappa=0.0):
    """Project points from 3D world coordinate system to 2D image.

    Parameters
    ----------
    P_world : array-like, shape (n_points, 4)
        Points in world coordinates

    cam2world : array-like, shape (4, 4), optional (default: I)
        Camera in world frame

    sensor_size : array-like, shape (2,)
        Size of the sensor array

    image_size : array-like, shape (2,)
        Size of the camera image: (width, height)

    focal_length : float
        Focal length of the camera

    image_center : array-like, shape (2,), optional (default: image_size / 2)
        Center of the image

    kappa : float, optional (default: 0)
        Camera distortion parameter

    Returns
    -------
    P_img : array-like, shape (n_points, 2)
        Points on image
    """
    world2cam = invert_transform(cam2world)
    P_cam = transform(world2cam, P_world)
    P_sensor = cam2sensor(P_cam, focal_length, kappa)
    P_img = sensor2img(P_sensor, sensor_size, image_size, image_center)
    return P_img


def plot_camera(ax=None, M=None, cam2world=None, virtual_image_distance=1.0,
                sensor_size=(1920, 1080), ax_s=1, strict_check=True,
                **kwargs):  # pragma: no cover
    """Plot camera in world coordinates.

    This function is inspired by Blender's camera visualization. It will
    show the camera center, a virtual image plane, and the top of the virtual
    image plane.

    Parameters
    ----------
    ax : Matplotlib 3d axis, optional (default: None)
        If the axis is None, a new 3d axis will be created.

    M : array-like, shape (3, 3)
        Intrinsic camera matrix that contains the focal lengths on the diagonal
        and the center of the the image in the last column. It does not matter
        whether values are given in meters or pixels as long as the unit is the
        same as for the sensor size.

    cam2world : array-like, shape (4, 4), optional (default: I)
        Transformation matrix of camera in world frame. We assume that the
        position is given in meters.

    virtual_image_distance : float, optional (default: 1)
        Distance from pinhole to virtual image plane that will be displayed.
        We assume that this distance is given in meters. The unit has to be
        consistent with the unit of the position in cam2world.

    sensor_size : array-like, shape (2,), optional (default: [1920, 1080])
        Size of the image sensor: (width, height). It does not matter whether
        values are given in meters or pixels as long as the unit is the same as
        for the sensor size.

    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis.

    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha.

    Returns
    -------
    ax : Matplotlib 3d axis
        New or old axis

    Raises
    ------
    ValueError
        If input is not valid
    """
    if ax is None:
        from .plot_utils import make_3d_axis
        ax = make_3d_axis(ax_s)

    if M is None:
        raise ValueError("No intrinsic camera matrix given.")

    if cam2world is None:
        cam2world = np.eye(4)

    cam2world = check_transform(cam2world, strict_check=strict_check)

    camera_center_in_world = cam2world[:3, 3]
    focal_length = np.mean(np.diag(M[:2, :2]))
    sensor_corners_in_cam = np.array([
        [0, 0, focal_length],
        [0, sensor_size[1], focal_length],
        [sensor_size[0], sensor_size[1], focal_length],
        [sensor_size[0], 0, focal_length],
    ])
    sensor_corners_in_cam[:, 0] -= M[0, 2]
    sensor_corners_in_cam[:, 1] -= M[1, 2]
    sensor_corners_in_world = transform(
        cam2world, vectors_to_points(sensor_corners_in_cam))[:, :3]
    virtual_image_corners = (
        virtual_image_distance / focal_length *
        (sensor_corners_in_world - camera_center_in_world[np.newaxis]) +
        camera_center_in_world[np.newaxis])

    if "c" in kwargs:
        color = kwargs.pop("c")
    elif "color" in kwargs:
        color = kwargs.pop("color")
    else:
        color = "k"

    _make_camera_frame(
        ax, virtual_image_corners, camera_center_in_world, color, **kwargs)
    _make_camera_top(ax, virtual_image_corners, color, **kwargs)
    ax.scatter(camera_center_in_world[0], camera_center_in_world[1],
               camera_center_in_world[2], color=color, **kwargs)

    return ax


def _make_camera_frame(
        ax, virtual_image_corners, camera_center_in_world, color,
        **kwargs):  # pragma: no cover
    """Plot camera frame.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        Axis

    virtual_image_corners : array, shape (4, 3)
        Virtual image corners in world coordinates.

    camera_center_in_world : array, shape (3,)
        Camera center in world coordinates.

    color : str
        Line color.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha.
    """
    camera_frame = np.vstack((
        virtual_image_corners[0],
        camera_center_in_world,
        virtual_image_corners[0],
        virtual_image_corners[1],
        camera_center_in_world,
        virtual_image_corners[1],
        virtual_image_corners[2],
        camera_center_in_world,
        virtual_image_corners[2],
        virtual_image_corners[3],
        camera_center_in_world,
        virtual_image_corners[3],
        virtual_image_corners[0],
    ))
    ax.plot(camera_frame[:, 0], camera_frame[:, 1], camera_frame[:, 2],
            color=color, **kwargs)


def _make_camera_top(
        ax, virtual_image_corners, color, **kwargs):  # pragma: no cover
    """Plot top indicator of camera.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        Axis

    virtual_image_corners : array, shape (4, 3)
        Virtual image corners in world coordinates.

    color : str
        Line color.

    kwargs : dict, optional (default: {})
        Additional arguments for the plotting functions, e.g. alpha.
    """
    up = virtual_image_corners[0] - virtual_image_corners[1]
    camera_top = np.array([
        virtual_image_corners[0] + 0.1 * up,
        0.5 * (virtual_image_corners[0] + virtual_image_corners[3]) + 0.5 * up,
        virtual_image_corners[3] + 0.1 * up,
        virtual_image_corners[0] + 0.1 * up
    ])
    ax.plot(camera_top[:, 0], camera_top[:, 1], camera_top[:, 2],
            color=color, **kwargs)
