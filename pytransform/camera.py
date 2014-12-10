import numpy as np
from .transformations import invert_transform, transform


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
    world_grid_x = np.vstack(
        [np.array([np.linspace(xlim[0], xlim[1], n_points_per_line),
                   np.linspace(y, y, n_points_per_line),
                   np.zeros(n_points_per_line),
                   np.ones(n_points_per_line)]).T
         for y in np.linspace(ylim[0], ylim[1], n_lines)])
    world_grid_y = np.vstack(
        [np.array([np.linspace(x, x, n_points_per_line),
                   np.linspace(ylim[0], ylim[1], n_points_per_line),
                   np.zeros(n_points_per_line),
                   np.ones(n_points_per_line)]).T
         for x in np.linspace(xlim[0], xlim[1], n_lines)])
    return np.vstack((world_grid_x, world_grid_y))


def cam2sensor(P_cam, focal_length, kappa=0.0):
    """Project points from 3D camera coordinate system to sensor plane.

    TODO document me
    """
    P_sensor = P_cam[:, :2] / P_cam[:, 2, np.newaxis]
    for n in range(P_sensor.shape[0]):
        P_sensor[n] *= 1.0 / (1.0 + kappa * np.linalg.norm(P_sensor[n]) ** 2)
    P_sensor *= focal_length
    return P_sensor


def sensor2img(P_sensor, sensor_size, image_size, image_center=None):
    """Project points from 2D sensor plane to image coordinate system.

    TODO document me
    """
    P_img = np.asarray(image_size) * P_sensor / np.asarray(sensor_size)
    if image_center is None:
        image_center = np.asarray(image_size) / 2
    P_img += np.asarray(image_center)
    return P_img


def world2image(P_world, cam2world, sensor_size, image_size, focal_length,
                image_center=None, kappa=0.0):
    """Project points from 3D world coordinate system to 2D image.

    TODO document me
    """
    world2cam = invert_transform(cam2world)
    P_cam = transform(world2cam, P_world)
    P_sensor = cam2sensor(P_cam, focal_length, kappa)
    P_img = sensor2img(P_sensor, sensor_size, image_size, image_center)
    return P_img
