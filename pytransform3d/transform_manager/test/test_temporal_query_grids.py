import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytransform3d.transform_manager import (
    NumpyTimeseriesTransform,
    TemporalTransformManager,
)


@pytest.mark.parametrize("shape", [(2, 3), (2, 1, 3), (0, 3)])
@pytest.mark.parametrize("clipping", [False, True])
def test_query_grid_matches_flat_queries(shape, clipping):
    pqs = np.array([[0, 0, 0, 1, 0, 0, 0], [2, 4, 6, 1, 0, 0, 0]], float)
    series = NumpyTimeseriesTransform([0, 1], pqs, time_clipping=clipping)
    queries = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    result = series.as_matrix(queries)
    assert result.shape == shape + (4, 4)
    assert_allclose(result.reshape(-1, 4, 4), series.as_matrix(queries.ravel()))
    assert_allclose(result[..., :3, 3], queries[..., None] * [2, 4, 6])


def test_query_grid_out_of_range_reports_only_invalid_times():
    pqs = np.array([[0, 0, 0, 1, 0, 0, 0], [2, 4, 6, 1, 0, 0, 0]], float)
    series = NumpyTimeseriesTransform([0, 1], pqs)
    with pytest.raises(
        ValueError, match=r"indices \[1 3\], time\(s\): \[-2\.  3\.\]"
    ):
        series.as_matrix(np.array([[0, -2], [0.5, 3]]))


def test_temporal_manager_query_grid_composes_dynamic_transforms():
    pqs = np.array([[0, 0, 0, 1, 0, 0, 0], [2, 4, 6, 1, 0, 0, 0]], float)
    manager = TemporalTransformManager()
    manager.add_transform(
        "camera", "body", NumpyTimeseriesTransform([0, 1], pqs)
    )
    manager.add_transform(
        "body", "world", NumpyTimeseriesTransform([0, 1], pqs)
    )
    queries = np.array([[0, 0.2], [0.6, 1]])
    result = manager.get_transform_at_time("camera", "world", queries)
    assert_allclose(result[..., :3, 3], queries[..., None] * [4, 8, 12])
