"""Optional 3D renderer based on Viser."""

import warnings

try:
    import open3d as o3d  # noqa: F401
    from ._figure import figure, Figure

    __all__ = [
        "figure",
        "Figure",
    ]
except ImportError:
    warnings.warn(
        "Viser is not available. Install viser.",
        ImportWarning,
        stacklevel=2,
    )
