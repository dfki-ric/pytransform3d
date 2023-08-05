import numpy as np

import pytransform3d._geometry as pg
import pytransform3d.transformations as pt
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_unit_sphere():
    x, y, z = pg.unit_sphere_surface_grid(10)
    assert_array_equal(x.shape, (10, 10))
    assert_array_equal(y.shape, (10, 10))
    assert_array_equal(z.shape, (10, 10))

    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    norms = np.linalg.norm(P, axis=1)
    assert_array_almost_equal(norms, np.ones_like(norms))


def test_transform_surface():
    x, y, z = pg.unit_sphere_surface_grid(10)

    p = np.array([0.2, -0.5, 0.7])
    pose = pt.transform_from(R=np.eye(3), p=p)
    x, y, z = pg.transform_surface(pose, x, y, z)

    P = np.column_stack((x.reshape(-1), y.reshape(-1), z.reshape(-1)))
    norms = np.linalg.norm(P - p[np.newaxis], axis=1)
    assert_array_almost_equal(norms, np.ones_like(norms))
