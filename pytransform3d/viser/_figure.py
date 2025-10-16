import numpy as np
import viser

from .. import rotations as pr
from .. import transformations as pt


class Figure:
    def __init__(self):
        self.server = viser.ViserServer()
        self.server.scene.configure_default_lights()
        self.grid_handle = self.server.scene.add_grid(
            name="grid", width=10, height=10
        )
        self.object_names = {}

    def plot_transform(self, A2B=None, s=1.0, name=None, strict_check=True):
        """Plot coordinate frame.

        Parameters
        ----------
        A2B : array-like, shape (4, 4)
            Transform from frame A to frame B

        s : float, optional (default: 1)
            Length of basis vectors

        name : str, optional (default: None)
            Name of the frame

        strict_check : bool, optional (default: True)
            Raise a ValueError if the transformation matrix is not
            numerically close enough to a real transformation matrix.
            Otherwise we print a warning.

        Returns
        -------
        Frame : frame
            New frame.
        """
        if "frame" not in self.object_names:
            self.object_names["frame"] = []
        if name is None:
            name = "/frame%05d" % len(self.object_names["frame"])

        if A2B is None:
            A2B = np.eye(4)
        A2B = pt.check_transform(A2B, strict_check=strict_check)

        frame = self.server.scene.add_frame(
            name,
            wxyz=pr.quaternion_from_matrix(
                A2B[:3, :3], strict_check=strict_check
            ),
            position=A2B[:3, 3],
        )

        return frame

    def plot_box(self, size=np.ones(3), A2B=np.eye(4), c=None):
        """Plot box.

        Parameters
        ----------
        size : array-like, shape (3,), optional (default: [1, 1, 1])
            Size of the box per dimension

        A2B : array-like, shape (4, 4), optional (default: I)
            Center of the box

        c : array-like, shape (3,), optional (default: None)
            Color

        Returns
        -------
        box : Box
            New box.
        """
        if "box" not in self.object_names:
            self.object_names["box"] = []

        name = "/box%05d" % len(self.object_names["box"])
        box = self.server.scene.add_box(
            name=name,
            dimensions=size,
            color=c,
            position=A2B[:3, 3],  # TODO orientation?
        )
        self.object_names["box"].append(name)
        return box

    def show(self):
        pass


def figure():
    return Figure()
