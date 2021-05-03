"""Layout utilities for matplotlib."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def make_3d_axis(ax_s, pos=111, unit=None, n_ticks=5):
    """Generate new 3D axis.

    Parameters
    ----------
    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    pos : int, optional (default: 111)
        Position indicator (nrows, ncols, plot_number)

    unit : str, optional (default: None)
        Unit of axes. For example, 'm', 'cm', 'km', ...
        The unit will be shown in the axis label, for example,
        as 'X [m]'.

    n_ticks : int, optional (default: 5)
        Number of ticks on each axis

    Returns
    -------
    ax : Matplotlib 3d axis
        New axis
    """
    try:
        ax = plt.subplot(pos, projection="3d", aspect="equal")
    except NotImplementedError:
        # HACK: workaround for bug in new matplotlib versions (ca. 3.02):
        # "It is not currently possible to manually set the aspect"
        ax = plt.subplot(pos, projection="3d")

    if unit is None:
        xlabel = "X"
        ylabel = "Y"
        zlabel = "Z"
    else:
        xlabel = "X [%s]" % unit
        ylabel = "Y [%s]" % unit
        zlabel = "Z [%s]" % unit

    plt.setp(
        ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
        xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

    ax.xaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(n_ticks))
    ax.zaxis.set_major_locator(MaxNLocator(n_ticks))

    ax.w_xaxis.pane.set_color("white")
    ax.w_yaxis.pane.set_color("white")
    ax.w_zaxis.pane.set_color("white")

    return ax


def remove_frame(ax, left=0.0, bottom=0.0, right=1.0, top=1.0):
    """Remove axis and scale bbox.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        Axis from which we remove the frame

    left : float, optional (default: 0)
        Position of left border (between 0 and 1)

    bottom : float, optional (default: 0)
        Position of bottom border (between 0 and 1)

    right : float, optional (default: 1)
        Position of right border (between 0 and 1)

    top : float, optional (default: 1)
        Position of top border (between 0 and 1)
    """
    ax.axis("off")
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
