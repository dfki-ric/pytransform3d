"""Manage compley chains of transformations.

See :doc:`transform_manager` for more information.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
try:
    import pydot
    pydot_available = True
except ImportError:
    pydot_available = False
from .transformations import (check_transform, invert_transform, concat,
                              plot_transform)
from .plot_utils import make_3d_axis


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

    .. warning::

        It is possible to introduce inconsistencies in the transform manager.
        Adding A2B and B2A with inconsistent values will result in an invalid
        state because inconsistencies will not be checked. It seems to be
        trivial in this simple case but can be computationally complex for
        large graphs. You can check the consistency explicitly with
        :func:`TransformManager.check_consistency`.

    Parameters
    ----------
    strict_check : bool, optional (default: True)
        Raise a ValueError if the transformation matrix is not numerically
        close enough to a real transformation matrix. Otherwise we print a
        warning.

    check : bool, optional (default: True)
        Check if transformation matrices are valid and requested nodes exist,
        which might significantly slow down some operations.
    """
    def __init__(self, strict_check=True, check=True):
        self.strict_check = strict_check
        self.check = check

        self.transforms = {}
        self.nodes = []
        self.i = []
        self.j = []
        self.connections = sp.csr_matrix((0, 0))
        self.dist = None
        self.predecessors = None
        self._cached_shortest_paths = {}

    def add_transform(self, from_frame, to_frame, A2B):
        """Register a transform.

        Parameters
        ----------
        from_frame : str
            Name of the frame for which the transform is added in the to_frame
            coordinate system

        to_frame : str
            Name of the frame in which the transform is defined

        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transform from 'from_frame'
            to 'to_frame'

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        if self.check:
            A2B = check_transform(A2B, strict_check=self.strict_check)
        if from_frame not in self.nodes:
            self.nodes.append(from_frame)
        if to_frame not in self.nodes:
            self.nodes.append(to_frame)

        recompute_shortest_path = False
        if (from_frame, to_frame) not in self.transforms:
            self.i.append(self.nodes.index(from_frame))
            self.j.append(self.nodes.index(to_frame))
            recompute_shortest_path = True

        self.transforms[(from_frame, to_frame)] = A2B

        if recompute_shortest_path:
            self._recompute_shortest_path()

        return self

    def _recompute_shortest_path(self):
        n_nodes = len(self.nodes)
        self.connections = sp.csr_matrix(
            (np.zeros(len(self.i)), (self.i, self.j)),
            shape=(n_nodes, n_nodes))
        self.dist, self.predecessors = csgraph.shortest_path(
            self.connections, unweighted=True, directed=False, method="D",
            return_predecessors=True)
        self._cached_shortest_paths.clear()

    def has_frame(self, frame):
        """Check if frame has been registered.

        Parameters
        ----------
        frame : str
            Frame name

        Returns
        -------
        has_frame : bool
            Frame is registered
        """
        return frame in self.nodes

    def get_transform(self, from_frame, to_frame):
        """Request a transform.

        Parameters
        ----------
        from_frame : str
            Name of the frame for which the transform is requested in the
            to_frame coordinate system

        to_frame : str
            Name of the frame in which the transform is defined

        Returns
        -------
        A2B : array-like, shape (4, 4)
            Homogeneous matrix that represents the transform from 'from_frame'
            to 'to_frame'
        """
        if self.check:
            if from_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % from_frame)
            if to_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % to_frame)

        if (from_frame, to_frame) in self.transforms:
            return self.transforms[(from_frame, to_frame)]
        elif (to_frame, from_frame) in self.transforms:
            return invert_transform(
                self.transforms[(to_frame, from_frame)],
                strict_check=self.strict_check, check=self.check)
        else:
            i = self.nodes.index(from_frame)
            j = self.nodes.index(to_frame)
            if not np.isfinite(self.dist[i, j]):
                raise KeyError("Cannot compute path from frame '%s' to "
                               "frame '%s'." % (from_frame, to_frame))

            path = self._shortest_path(i, j)
            return self._path_transform(path)

    def _shortest_path(self, i, j):
        if (i, j) in self._cached_shortest_paths:
            return self._cached_shortest_paths[(i, j)]

        path = []
        k = i
        while k != -9999:
            path.append(self.nodes[k])
            k = self.predecessors[j, k]
        self._cached_shortest_paths[(i, j)] = path
        return path

    def _path_transform(self, path):
        A2B = np.eye(4)
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B = concat(A2B, self.get_transform(from_f, to_f),
                         strict_check=self.strict_check, check=self.check)
        return A2B

    def plot_frames_in(self, frame, ax=None, s=1.0, ax_s=1, show_name=True, whitelist=None, **kwargs):
        """Plot all frames in a given reference frame.

        Note that frames that cannot be connected to the reference frame are
        omitted.

        Parameters
        ----------
        frame : str
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        s : float, optional (default: 1)
            Scaling of the frame that will be drawn

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        show_name : bool, optional (default: True)
            Print node names

        whitelist : list, optional (default: None)
            Frames that must be plotted

        kwargs : dict, optional (default: {})
            Additional arguments for the plotting functions, e.g. alpha

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if frame not in self.nodes:
            raise KeyError("Unknown frame '%s'" % frame)

        nodes = self._whitelisted_nodes(whitelist)

        for node in nodes:
            try:
                node2frame = self.get_transform(node, frame)
                name = node if show_name else None
                ax = plot_transform(
                    ax, node2frame, s, ax_s, name,
                    strict_check=self.strict_check, **kwargs)
            except KeyError:
                pass  # Frame is not connected to the reference frame
        return ax

    def plot_connections_in(self, frame, ax=None, ax_s=1, whitelist=None, **kwargs):
        """Plot direct frame connections in a given reference frame.

        A line between each pair of frames for which a direct transformation
        is known will be plotted. Direct means that either A2B or B2A has been
        added to the transform manager.

        Note that frames that cannot be connected to the reference frame are
        omitted.

        Parameters
        ----------
        frame : str
            Reference frame

        ax : Matplotlib 3d axis, optional (default: None)
            If the axis is None, a new 3d axis will be created

        ax_s : float, optional (default: 1)
            Scaling of the new matplotlib 3d axis

        whitelist : list, optional (default: None)
            Both frames of a connection must be in the whitelist to plot the
            connection

        kwargs : dict, optional (default: {})
            Additional arguments for the plotting functions, e.g. alpha

        Returns
        -------
        ax : Matplotlib 3d axis
            New or old axis
        """
        if frame not in self.nodes:
            raise KeyError("Unknown frame '%s'" % frame)

        if ax is None:
            ax = make_3d_axis(ax_s)

        nodes = self._whitelisted_nodes(whitelist)

        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = "black"

        for frame_names, transform in self.transforms.items():
            from_frame, to_frame = frame_names
            if from_frame in nodes and to_frame in nodes:
                try:
                    from2ref = self.get_transform(from_frame, frame)
                    to2ref = self.get_transform(to_frame, frame)
                    ax.plot(
                        (from2ref[0, 3], to2ref[0, 3]),
                        (from2ref[1, 3], to2ref[1, 3]),
                        (from2ref[2, 3], to2ref[2, 3]),
                        **kwargs
                    )
                except KeyError:
                    pass  # Frame is not connected to the reference frame

        return ax

    def _whitelisted_nodes(self, whitelist):
        """Get whitelisted nodes.

        A KeyError will be raised if an unknown node is in the whitelist.

        Parameters
        ----------
        whitelist : list or None
            Whitelist of frames

        Returns
        -------
        nodes : set
            Existing whitelisted nodes
        """
        nodes = set(self.nodes)
        if whitelist is not None:
            whitelist = set(whitelist)
            nodes = nodes.intersection(whitelist)
            nonwhitlisted_nodes = whitelist.difference(nodes)
            if nonwhitlisted_nodes:
                raise KeyError("Whitelist contains unknown nodes: '%s'"
                               % nonwhitlisted_nodes)
        return nodes

    def check_consistency(self):
        """Check consistency of the known transformations.

        The complexity of this is between :math:`O(n^2)` and :math:`O(n^3)`,
        where :math:`n` is the number of nodes. In graphs where each pair of
        nodes is directly connected the complexity is :math:`O(n^2)`. In graphs
        that are actually paths, the complexity is :math:`O(n^3)`.

        Returns
        -------
        consistent : bool
            Is the graph consistent, i.e. is A2B always the same as the inverse
            of B2A?
        """
        consistent = True
        for n1 in self.nodes:
            for n2 in self.nodes:
                try:
                    n1_to_n2 = self.get_transform(n1, n2)
                    n2_to_n1 = self.get_transform(n2, n1)
                    n1_to_n2_inv = invert_transform(
                        n2_to_n1, strict_check=self.strict_check,
                        check=self.check)
                    consistent = (consistent and
                                  np.allclose(n1_to_n2, n1_to_n2_inv))
                except KeyError:
                    pass  # Frames are not connected
        return consistent

    def connected_components(self):
        """Get number of connected components.

        If the number is larger than 1 there will be frames without
        connections.

        Returns
        -------
        n_connected_components : int
            Number of connected components.
        """
        return csgraph.connected_components(
            self.connections, directed=False, return_labels=False)

    def write_png(self, filename, prog=None):
        """Create PNG from dot graph of the transformations.

        .. warning::

            Note that this method requires the Python package pydot and an
            existing installation of graphviz on your system.

        Parameters
        ----------
        filename : str
            Name of the output file. Should end with '.png'.

        prog : str, optional (default: dot)
            Name of GraphViz executable that can be found in the `$PATH` or
            absolute path to GraphViz executable. Possible options are, for
            example, 'dot', 'twopi', 'neato', 'circo', 'fdp', 'sfdp'.
        """
        if not pydot_available:
            raise ImportError("pydot must be installed to use this feature.")

        graph = pydot.Dot(graph_type="graph")
        frame_color = "#dd3322"
        connection_color = "#d0d0ff"

        for frame in self.nodes:
            node = pydot.Node(
                self.__display_name(frame), style="filled",
                fillcolor=frame_color, shape="egg")
            graph.add_node(node)
        for frames, A2B in self.transforms.items():
            a, b = frames
            connection_name = "%s to %s\n%s" % (
                self.__display_name(a), self.__display_name(b),
                str(np.round(A2B, 3)))
            node = pydot.Node(
                connection_name, style="filled", fillcolor=connection_color,
                shape="note")
            graph.add_node(node)
            a_name = self.__display_name(a)
            a_edge = pydot.Edge(connection_name, a_name, penwidth=3)
            graph.add_edge(a_edge)
            b_name = self.__display_name(b)
            b_edge = pydot.Edge(connection_name, b_name, penwidth=3)
            graph.add_edge(b_edge)

        graph.write_png(filename, prog=prog)

    def __display_name(self, name):
        return name.replace("/", "")
