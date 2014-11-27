import numpy as np
from .transformations import invert_transform, transform


def world2image(cam2world, P_world, center_image, focal_length, kappa):
    """TODO document me"""
    center_image = np.asarray(center_image)

    world2cam = invert_transform(cam2world)
    P_cam = transform(world2cam, P_world)

    P_img = np.empty((P_cam.shape[0], 2))
    for n in range(len(P_img)):
        # Project to camera image
        P_img[n] = P_cam[n, :2] / P_cam[n, 2]
        # Remove distortion
        P_img[n] *= 1.0 / (1.0 + kappa * np.linalg.norm(P_img[n]) ** 2)
        P_img[n] *= focal_length
        P_img[n] += center_image

    return P_img
