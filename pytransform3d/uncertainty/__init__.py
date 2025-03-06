"""Operations related to uncertain transformations.

See :doc:`user_guide/uncertainty` for more information.
"""

from ._composition import (
    concat_globally_uncertain_transforms,
    concat_locally_uncertain_transforms,
)
from ._frechet_mean import (
    estimate_gaussian_rotation_matrix_from_samples,
    estimate_gaussian_transform_from_samples,
    frechet_mean,
)
from ._fusion import pose_fusion
from ._invert import invert_uncertain_transform
from ._plot import (
    to_ellipsoid,
    to_projected_ellipsoid,
    plot_projected_ellipsoid,
)

__all__ = [
    "concat_globally_uncertain_transforms",
    "concat_locally_uncertain_transforms",
    "frechet_mean",
    "estimate_gaussian_rotation_matrix_from_samples",
    "estimate_gaussian_transform_from_samples",
    "pose_fusion",
    "invert_uncertain_transform",
    "to_ellipsoid",
    "to_projected_ellipsoid",
    "plot_projected_ellipsoid",
]
