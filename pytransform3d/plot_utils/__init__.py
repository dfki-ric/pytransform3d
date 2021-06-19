"""Utilities for plotting."""
import warnings
try:
    import matplotlib.pyplot as plt
    from ._artists import Arrow3D, Frame, LabeledFrame, Trajectory
    from ._layout import make_3d_axis, remove_frame
    from ._plot_functions import (
        plot_box, plot_sphere, plot_cylinder, plot_mesh, plot_ellipsoid,
        plot_capsule, plot_cone, plot_vector, plot_length_variable)

    __all__ = [
        "Arrow3D", "Frame", "LabeledFrame", "Trajectory",
        "make_3d_axis", "remove_frame",
        "plot_box", "plot_sphere", "plot_cylinder", "plot_mesh",
        "plot_ellipsoid", "plot_capsule", "plot_cone", "plot_vector",
        "plot_length_variable"
    ]
except ImportError:
    warnings.warn("Matplotlib is not installed, visualization is not "
                  "available")
