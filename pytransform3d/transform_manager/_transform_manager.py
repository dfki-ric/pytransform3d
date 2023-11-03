import numpy as np
import scipy.sparse as sp

from ._transform_graph_base import TransformGraphBase
from ..transformations import check_transform, plot_transform

try:  # pragma: no cover
    import pydot
    PYDOT_AVAILABLE = True
except ImportError:
    PYDOT_AVAILABLE = False


class TransformManager(TransformGraphBase):
    """Manage transformations between frames.

    This is a simplified version of `ROS tf <http://wiki.ros.org/tf>`_ that
    ignores the temporal aspect. A user can register transformations. The
    shortest path between all frames will be computed internally which enables
    us to provide transforms for any connected frames.

    Suppose we know the transformations A2B, D2C, and B2C. The transform
    manager can compute any transformation between the frames A, B, C and D.
    For example, you can request the transformation that represents frame D in
    frame A. The transformation manager will automatically concatenate the
    transformations D2C, C2B, and B2A, where C2B and B2A are obtained by
    inverting B2C and A2B respectively.

    .. warning::

        It is possible to introduce inconsistencies in the transformation
        manager. Adding A2B and B2A with inconsistent values will result in
        an invalid state because inconsistencies will not be checked. It seems
        to be trivial in this simple case but can be computationally complex
        for large graphs. You can check the consistency explicitly with
        :func:`TransformManager.check_consistency`.

    The TransformManager does not directly support serialization because
    we don't want to decide for a specific format. However, it allows
    conversion to a dict with only primitive types that is serializable,
    for instance, as JSON. If a more compact format is required, binary
    formats like msgpack can be used. Use :func:`TransformManager.to_dict`
    and :func:`TransformManager.from_dict` for this purpose.

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
        super(TransformManager, self).__init__(strict_check, check)
        self._transforms = {}

    @property
    def transforms(self):
        """Rigid transformations between nodes."""
        return self._transforms

    def _check_transform(self, A2B):
        """Check validity of rigid transformation."""
        return check_transform(A2B, strict_check=self.strict_check)

    def _transform_available(self, key):
        return key in self._transforms

    def _set_transform(self, key, A2B):
        self._transforms[key] = A2B

    def _get_transform(self, key):
        return self._transforms[key]

    def _del_transform(self, key):
        del self._transforms[key]

    def plot_frames_in(self, frame, ax=None, s=1.0, ax_s=1, show_name=True,
                       whitelist=None, **kwargs):  # pragma: no cover
        """Plot all frames in a given reference frame.

        Note that frames that cannot be connected to the reference frame are
        omitted.

        Parameters
        ----------
        frame : Hashable
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

        Raises
        ------
        KeyError
            If the frame is unknown
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

    def plot_connections_in(self, frame, ax=None, ax_s=1, whitelist=None,
                            **kwargs):  # pragma: no cover
        """Plot direct frame connections in a given reference frame.

        A line between each pair of frames for which a direct transformation
        is known will be plotted. Direct means that either A2B or B2A has been
        added to the transformation manager.

        Note that frames that cannot be connected to the reference frame are
        omitted.

        Parameters
        ----------
        frame : Hashable
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

        Raises
        ------
        KeyError
            If the frame is unknown
        """
        if frame not in self.nodes:
            raise KeyError("Unknown frame '%s'" % frame)

        if ax is None:
            from ..plot_utils import make_3d_axis
            ax = make_3d_axis(ax_s)

        nodes = self._whitelisted_nodes(whitelist)

        if "c" not in kwargs and "color" not in kwargs:
            kwargs["color"] = "black"

        for frame_names in self._transforms:
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

        Parameters
        ----------
        whitelist : list or None
            Whitelist of frames

        Returns
        -------
        nodes : set
            Existing whitelisted nodes

        Raises
        ------
        KeyError
            Will be raised if an unknown node is in the whitelist.
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

    def write_png(self, filename, prog=None):  # pragma: no cover
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

        Raises
        ------
        ImportError
            If pydot is not available
        """
        if not PYDOT_AVAILABLE:
            raise ImportError("pydot must be installed to use this feature.")

        graph = pydot.Dot(graph_type="graph")
        frame_color = "#dd3322"
        connection_color = "#d0d0ff"

        for frame in self.nodes:
            node = pydot.Node(
                _dot_display_name(str(frame)), style="filled",
                fillcolor=frame_color, shape="egg")
            graph.add_node(node)
        for frames, A2B in self._transforms.items():
            frame_a, frame_b = frames
            connection_name = "%s to %s\n%s" % (
                _dot_display_name(str(frame_a)),
                _dot_display_name(str(frame_b)), str(np.round(A2B, 3)))
            node = pydot.Node(
                connection_name, style="filled", fillcolor=connection_color,
                shape="note")
            graph.add_node(node)
            a_name = _dot_display_name(str(frame_a))
            a_edge = pydot.Edge(connection_name, a_name, penwidth=3)
            graph.add_edge(a_edge)
            b_name = _dot_display_name(str(frame_b))
            b_edge = pydot.Edge(connection_name, b_name, penwidth=3)
            graph.add_edge(b_edge)

        graph.write_png(filename, prog=prog)

    def to_dict(self):
        """Convert the transform manager to a dict that is serializable.

        Returns
        -------
        tm_dict : dict
            Serializable dict.
        """
        return {
            "class": self.__class__.__name__,
            "strict_check": self.strict_check,
            "check": self.check,
            "transforms": [(k, v.ravel().tolist())
                           for k, v in self._transforms.items()],
            "nodes": self.nodes,
            "i": self.i,
            "j": self.j,
            "transform_to_ij_index": list(self.transform_to_ij_index.items()),
            "connections": {
                "data": self.connections.data.tolist(),
                "indices": self.connections.indices.tolist(),
                "indptr": self.connections.indptr.tolist()
            },
            "dist": self.dist.tolist(),
            "predecessors": self.predecessors.tolist()
        }

    @staticmethod
    def from_dict(tm_dict):
        """Create transform manager from dict.

        Parameters
        ----------
        tm_dict : dict
            Serializable dict.

        Returns
        -------
        tm : TransformManager
            Deserialized transform manager.
        """
        strict_check = tm_dict.get("strict_check")
        check = tm_dict.get("check")
        tm = TransformManager(strict_check=strict_check, check=check)
        tm.set_transform_manager_state(tm_dict)
        return tm

    def set_transform_manager_state(self, tm_dict):
        """Set state of transform manager from dict.

        Parameters
        ----------
        tm_dict : dict
            Serializable dict.
        """
        transforms = tm_dict.get("transforms")
        self._transforms = {tuple(k): np.array(v).reshape(4, 4)
                            for k, v in transforms}
        self.nodes = tm_dict.get("nodes")
        self.i = tm_dict.get("i")
        self.j = tm_dict.get("j")
        self.transform_to_ij_index = dict(
            (tuple(k), v) for k, v in tm_dict.get("transform_to_ij_index"))
        connections = tm_dict.get("connections")
        self.connections = sp.csr_matrix((
            connections["data"], connections["indices"],
            connections["indptr"]))
        n_nodes = len(self.nodes)
        dist = np.array(tm_dict.get("dist"))
        self.dist = dist.reshape(n_nodes, n_nodes)
        predecessors = np.array(tm_dict.get("predecessors"), dtype=np.int32)
        self.predecessors = predecessors.reshape(n_nodes, n_nodes)


def _dot_display_name(name):  # pragma: no cover
    return name.replace("/", "")
