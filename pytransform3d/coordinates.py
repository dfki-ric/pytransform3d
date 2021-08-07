"""Conversions between coordinate systems to represent positions."""
import numpy as np


def cartesian_from_cylindrical(p):
    """Convert cylindrical coordinates to Cartesian coordinates.

    Parameters
    ----------
    p : array-like, shape (..., 3)
        Cylindrical coordinates: axial / radial distance (rho), azimuth
        (phi), and axial coordinate / height (z)

    Returns
    -------
    q : array, shape (..., 3)
        Cartesian coordinates (x, y, z)
    """
    p = np.asarray(p)
    q = np.empty_like(p)
    q[..., 0] = p[..., 0] * np.cos(p[..., 1])
    q[..., 1] = p[..., 0] * np.sin(p[..., 1])
    q[..., 2] = p[..., 2]
    return q


def cartesian_from_spherical(p):
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    p : array-like, shape (..., 3)
        Spherical coordinates: radial distance (rho), inclination /
        elevation (theta), and azimuth (phi)

    Returns
    -------
    q : array, shape (..., 3)
        Cartesian coordinates (x, y, z)
    """
    p = np.asarray(p)
    q = np.empty_like(p)
    r_sin_theta = p[..., 0] * np.sin(p[..., 1])
    q[..., 0] = np.cos(p[..., 2]) * r_sin_theta
    q[..., 1] = np.sin(p[..., 2]) * r_sin_theta
    q[..., 2] = p[..., 0] * np.cos(p[..., 1])
    return q


def cylindrical_from_cartesian(p):
    """Convert Cartesian coordinates to cylindrical coordinates.

    Parameters
    ----------
    p : array-like, shape (..., 3)
        Cartesian coordinates (x, y, z)

    Returns
    -------
    q : array, shape (..., 3)
        Cylindrical coordinates: axial / radial distance (rho >= 0), azimuth
        (-pi >= phi >= pi), and axial coordinate / height (z)
    """
    p = np.asarray(p)
    q = np.empty_like(p)
    q[..., 0] = np.linalg.norm(p[..., :2], axis=-1)
    q[..., 1] = np.arctan2(p[..., 1], p[..., 0])
    q[..., 2] = p[..., 2]
    return q


def cylindrical_from_spherical(p):
    """Convert spherical coordinates to cylindrical coordinates.

    Parameters
    ----------
    p : array-like, shape (..., 3)
        Spherical coordinates: radial distance (rho), inclination /
        elevation (theta), and azimuth (phi)

    Returns
    -------
    q : array, shape (..., 3)
        Cylindrical coordinates: axial / radial distance (rho), azimuth
        (phi), and axial coordinate / height (z)
    """
    p = np.asarray(p)
    q = np.empty_like(p)
    q[..., 0] = p[..., 0] * np.sin(p[..., 1])
    q[..., 1] = p[..., 2]
    q[..., 2] = p[..., 0] * np.cos(p[..., 1])
    return q


def spherical_from_cartesian(p):
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    p : array-like, shape (..., 3)
        Cartesian coordinates (x, y, z)

    Returns
    -------
    q : array, shape (..., 3)
        Spherical coordinates: radial distance (rho >= 0), inclination /
        elevation (0 <= theta <= pi), and azimuth (-pi <= phi <= pi)
    """
    p = np.asarray(p)
    q = np.empty_like(p)
    q[..., 0] = np.linalg.norm(p, axis=-1)
    q[..., 1] = np.arctan2(np.linalg.norm(p[..., :2], axis=-1), p[..., 2])
    q[..., 2] = np.arctan2(p[..., 1], p[..., 0])
    return q


def spherical_from_cylindrical(p):
    """Convert cylindrical coordinates to spherical coordinates.

    Parameters
    ----------
    p : array-like, shape (..., 3)
        Cylindrical coordinates: axial / radial distance (rho), azimuth
        (phi), and axial coordinate / height (z)

    Returns
    -------
    q : array, shape (..., 3)
        Spherical coordinates: radial distance (rho), inclination /
        elevation (theta), and azimuth (phi)
    """
    p = np.asarray(p)
    q = np.empty_like(p)
    q[..., 0] = np.linalg.norm(p[..., (0, 2)], axis=-1)
    q[..., 1] = np.arctan2(p[..., 0], p[..., 2])
    q[..., 2] = p[..., 1]
    return q
