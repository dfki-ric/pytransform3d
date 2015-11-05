import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):  # http://stackoverflow.com/a/11156353/915743
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def make_3d_axis(ax_s):
    """Generate new 3D axis.

    Parameters
    ----------
    ax_s : float, optional (default: 1)
        Scaling of the new matplotlib 3d axis

    Returns
    -------
    ax : Matplotlib 3d axis
        New axis
    """
    ax = plt.subplot(111, projection="3d", aspect="equal")
    plt.setp(ax, xlim=(-ax_s, ax_s), ylim=(-ax_s, ax_s), zlim=(-ax_s, ax_s),
             xlabel="X", ylabel="Y", zlabel="Z")
    return ax
