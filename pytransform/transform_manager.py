import numpy as np
import scipy.sparse as sp
from .transformations import (check_transform, invert_transform, concat,
                              plot_transform)


class TransformManager(object):
    """Manage transforms between frames.

    This is a simplified version of `ROS tf <http://wiki.ros.org/tf>`_ that
    ignores the temporal aspect. A user can register transforms. The shortest
    path between all frames will be computed internally which enables us to
    provide transforms for any connected frames.

    Suppose we know the transforms A2B, D2C, and B2C. The transform manager can
    compute any transform between the frames A, B, C and D. For example, you
    can request the transform that represents frame D in frame A. The transform
    manager will automatically concatenate the transform D2C, C2B, and B2A,
    where C2B and B2A are obtained by inverting B2C and A2B respectively.
    """
    def __init__(self):
        self.transforms = {}
        self.nodes = []
        self.i = []
        self.j = []

    def add_transform(self, from_frame, to_frame, A2B):
        """Register a transform.

        Parameters
        ----------
        from_frame : string
            Name of the frame for which the transform is added in the to_frame
            coordinate system

        to_frame : string
            Name of the frame in which the transform is defined

        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transform from 'from_frame'
            to 'to_frame'

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        A2B = check_transform(A2B)
        if from_frame not in self.nodes:
            self.nodes.append(from_frame)
        if to_frame not in self.nodes:
            self.nodes.append(to_frame)
        self.i.append(self.nodes.index(from_frame))
        self.j.append(self.nodes.index(to_frame))
        self.transforms[(from_frame, to_frame)] = A2B

        n_nodes = len(self.nodes)
        con = sp.csr_matrix((np.zeros(len(self.i)), (self.i, self.j)),
                            shape=(n_nodes, n_nodes))
        self.dist, self.predecessors = sp.csgraph.shortest_path(
            con, unweighted=True, return_predecessors=True, directed=False)

        return self

    def get_transform(self, from_frame, to_frame):
        """Request a transform.

        Parameters
        ----------
        from_frame : string
            Name of the frame for which the transform is requested in the
            to_frame coordinate system

        to_frame : string
            Name of the frame in which the transform is defined

        Returns
        -------
        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transform from 'from_frame'
            to 'to_frame'
        """
        if from_frame not in self.nodes:
            raise KeyError("Unknown frame '%s'" % from_frame)
        if to_frame not in self.nodes:
            raise KeyError("Unknown frame '%s'" % to_frame)

        if (from_frame, to_frame) in self.transforms:
            return self.transforms[(from_frame, to_frame)]
        elif (to_frame, from_frame) in self.transforms:
            return invert_transform(self.transforms[(to_frame, from_frame)])
        else:
            i = self.nodes.index(from_frame)
            j = self.nodes.index(to_frame)
            if not np.isfinite(self.dist[i, j]):
                raise KeyError("Cannot compute path from frame '%s' to "
                               "frame '%s'." % (from_frame, to_frame))

            path = self._shortest_path(i, j)
            return self._path_transform(path)

    def _shortest_path(self, i, j):
        path = []
        k = i
        while k != -9999:
            path.append(self.nodes[k])
            k = self.predecessors[j, k]
        return path

    def _path_transform(self, path):
        A2B = np.eye(4)
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B = concat(A2B, self.get_transform(from_f, to_f))
        return A2B

    def plot_frames_in(self, frame, ax=None, s=1.0, ax_s=1, **kwargs):
        """Plot all frames in a given reference frame.

        Note that frames that cannot be connected to the reference frame are
        omitted.

        Parameters
        ----------
        frame : string
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        s : float, optional (default: 1)
            Scaling of the axis and angle that will be drawn

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        kwargs : dict, optional (default: {})
            Additional arguments for the plotting functions, e.g. alpha

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if frame not in self.nodes:
            raise KeyError("Unknown frame '%s'" % from_frame)

        for node in self.nodes:
            try:
                node2frame = self.get_transform(node, frame)
                ax = plot_transform(ax, node2frame, s, ax_s, node, **kwargs)
            except KeyError:
                pass  # Frame is not connected to the reference frame
        return ax
