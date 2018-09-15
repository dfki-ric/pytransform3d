import matplotlib.pyplot as plt
from matplotlib import artist
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3D, Text3D


class Frame(artist.Artist):
    """A frame or coordinate system represented by its basis vectors.

    Parameters
    ----------
    A2B : array-like, shape (4, 4)
        Transform from frame A to frame B

    label : str, optional (default: None)
        Name of the frame

    s : float, optional (default: 1)
        Length of basis vectors

    Other arguments except 'c' and 'color' are passed onto Line3D.
    """
    def __init__(self, A2B, label=None, s=1.0, **kwargs):
        super(Frame, self).__init__()

        if "c" in kwargs:
            kwargs.pop("c")
        if "color" in kwargs:
            kwargs.pop("color")

        self.s = s

        self.x_axis = Line3D([], [], [], color="r", **kwargs)
        self.y_axis = Line3D([], [], [], color="g", **kwargs)
        self.z_axis = Line3D([], [], [], color="b", **kwargs)

        self.draw_label = label is not None
        self.label = label

        if self.draw_label:
            self.label_indicator = Line3D([], [], [], color="k", **kwargs)
            self.label_text = Text3D(0, 0, 0, text="", zdir="z")

        self.set_data(A2B, label)

    def set_data(self, A2B, label=None):
        """Set the transformation data.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        label : str, optional (default: None)
            Name of the frame
        """
        R = A2B[:3, :3]
        p = A2B[:3, 3]

        for d, b in enumerate([self.x_axis, self.y_axis, self.z_axis]):
            b.set_data([p[0], p[0] + self.s * R[0, d]],
                       [p[1], p[1] + self.s * R[1, d]])
            b.set_3d_properties([p[2], p[2] + self.s * R[2, d]])

        if self.draw_label:
            if label is None:
                label = self.label
            label_pos = p + 0.5 * self.s * (R[:, 0] + R[:, 1] + R[:, 2])

            self.label_indicator.set_data(
                [p[0], label_pos[0]], [p[1], label_pos[1]])
            self.label_indicator.set_3d_properties([p[2], label_pos[2]])

            self.label_text.set_text(label)
            self.label_text.set_position([label_pos[0], label_pos[1]])
            self.label_text.set_3d_properties(label_pos[2])

    @artist.allow_rasterization
    def draw(self, renderer):
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            b.draw(renderer)
        if self.draw_label:
            self.label_indicator.draw(renderer)
            self.label_text.draw(renderer)

    def add_frame(self, axis):
        for b in [self.x_axis, self.y_axis, self.z_axis]:
            axis.add_line(b)
        if self.draw_label:
            axis.add_line(self.label_indicator)
            axis._add_text(self.label_text)


class Arrow3D(FancyArrowPatch):  # http://stackoverflow.com/a/11156353/915743
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def make_3d_axis(ax_s, pos=111):
    """Generate new 3D axis.

    Parameters
    ----------
    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    pos : int, optional (default: 111)
        Position indicator (nrows, ncols, plot_number)

    Returns
    -------
    ax : Matplotlib 3d axis
        New axis
    """
    ax = plt.subplot(pos, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
             xlabel="X", ylabel="Y", zlabel="Z")
    return ax
