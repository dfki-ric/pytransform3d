import numpy as np
from .transformations import invert_transform, transform


def make_world_grid(n_lines=11, n_points_per_line=51, xlim=(-0.5, 0.5),
                    ylim=(-0.5, 0.5)):
    """TODO document me"""
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
    """TODO document me"""
    P_sensor = P_cam[:, :2] / -P_cam[:, 2, np.newaxis]
    for n in range(P_sensor.shape[0]):
        P_sensor[n] *= 1.0 / (1.0 + kappa * np.linalg.norm(P_sensor[n]) ** 2)
    P_sensor *= focal_length
    return P_sensor


def sensor2img(P_sensor, sensor_size, image_size, image_center=None):
    """TODO document me"""
    P_img = np.asarray(image_size) * P_sensor / np.asarray(sensor_size)
    if image_center is None:
        image_center = np.asarray(image_size) / 2
    P_img += np.asarray(image_center)
    # y-axis of image goes from top to bottom
    P_img[:, 1] = image_size[1] - P_img[:, 1]
    return P_img


def world2image(P_world, cam2world, sensor_size, image_size, focal_length,
                image_center=None, kappa=0.0):
    """TODO document me"""
    world2cam = invert_transform(cam2world)
    P_cam = transform(world2cam, P_world)
    P_sensor = cam2sensor(P_cam, focal_length, kappa)
    P_img = sensor2img(P_sensor, sensor_size, image_size, image_center)
    return P_img
