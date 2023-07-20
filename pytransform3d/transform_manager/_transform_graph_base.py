import abc

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph

from ..transformations import concat, invert_transform


class TransformGraphBase(abc.ABC):
    """Base class for all graphs of rigid transformations.

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

        # Names of nodes
        self.nodes = []

        # A pair (self.i[n], self.j[n]) represents indices of connected nodes
        self.i = []
        self.j = []
        # We have to store the index n associated to a transformation to be
        # able to remove the transformation later
        self.transform_to_ij_index = {}
        # Connection information as sparse matrix
        self.connections = sp.csr_matrix((0, 0))
        # Result of shortest path algorithm:
        # distance matrix (distance is the number of transformations)
        self.dist = np.empty(0)
        self.predecessors = np.empty(0, dtype=np.int32)

        self._cached_shortest_paths = {}

    @property
    @abc.abstractmethod
    def transforms(self):
        """Rigid transformations between nodes."""

    @abc.abstractmethod
    def _check_transform(self, A2B):
        """Check validity of rigid transformation."""

    def _path_transform(self, path):
        """Convert sequence of node names to rigid transformation."""
        A2B = np.eye(4)
        for from_f, to_f in zip(path[:-1], path[1:]):
            A2B = concat(A2B, self.get_transform(from_f, to_f),
                         strict_check=self.strict_check, check=self.check)
        return A2B

    @abc.abstractmethod
    def _transform_available(self, key):
        """Check if transformation key is available."""

    @abc.abstractmethod
    def _set_transform(self, key, A2B):
        """Store transformation under given key."""

    @abc.abstractmethod
    def _get_transform(self, key):
        """Retrieve stored transformation under given key."""

    @abc.abstractmethod
    def _del_transform(self, key):
        """Delete transformation stored under given key."""

    def has_frame(self, frame):
        """Check if frame has been registered.

        Parameters
        ----------
        frame : Hashable
            Frame name

        Returns
        -------
        has_frame : bool
            Frame is registered
        """
        return frame in self.nodes

    def add_transform(self, from_frame, to_frame, A2B):
        """Register a transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is added in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        A2B : Any
            Transformation from 'from_frame' to 'to_frame'

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        if self.check:
            A2B = self._check_transform(A2B)

        if from_frame not in self.nodes:
            self.nodes.append(from_frame)
        if to_frame not in self.nodes:
            self.nodes.append(to_frame)

        transform_key = (from_frame, to_frame)

        recompute_shortest_path = False
        if not self._transform_available(transform_key):
            ij_index = len(self.i)
            self.i.append(self.nodes.index(from_frame))
            self.j.append(self.nodes.index(to_frame))
            self.transform_to_ij_index[transform_key] = ij_index
            recompute_shortest_path = True

        if recompute_shortest_path:
            self._recompute_shortest_path()

        self._set_transform(transform_key, A2B)

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

    def remove_transform(self, from_frame, to_frame):
        """Remove a transformation.

        Nothing happens if there is no such transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is added in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        Returns
        -------
        self : TransformManager
            This object for chaining
        """
        transform_key = (from_frame, to_frame)
        if self._transform_available(transform_key):
            self._del_transform(transform_key)
            ij_index = self.transform_to_ij_index[transform_key]
            del self.transform_to_ij_index[transform_key]
            self.transform_to_ij_index = {
                k: v if v < ij_index else v - 1
                for k, v in self.transform_to_ij_index.items()}
            del self.i[ij_index]
            del self.j[ij_index]
            self._recompute_shortest_path()
        return self

    def get_transform(self, from_frame, to_frame):
        """Request a transformation.

        Parameters
        ----------
        from_frame : Hashable
            Name of the frame for which the transformation is requested in the
            to_frame coordinate system

        to_frame : Hashable
            Name of the frame in which the transformation is defined

        Returns
        -------
        A2B : Any
            Transformation from 'from_frame' to 'to_frame'

        Raises
        ------
        KeyError
            If one of the frames is unknown or there is no connection between
            them
        """
        if self.check:
            if from_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % from_frame)
            if to_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % to_frame)

        if self._transform_available((from_frame, to_frame)):
            return self._get_transform((from_frame, to_frame))

        if self._transform_available((to_frame, from_frame)):
            return invert_transform(
                self._get_transform((to_frame, from_frame)),
                strict_check=self.strict_check,
                check=self.check
            )

        i = self.nodes.index(from_frame)
        j = self.nodes.index(to_frame)
        if not np.isfinite(self.dist[i, j]):
            raise KeyError("Cannot compute path from frame '%s' to "
                           "frame '%s'." % (from_frame, to_frame))

        path = self._shortest_path(i, j)
        return self._path_transform(path)

    def _shortest_path(self, i, j):
        """Names of nodes along the shortest path between two indices."""
        if (i, j) in self._cached_shortest_paths:
            return self._cached_shortest_paths[(i, j)]

        path = []
        k = i
        while k != -9999:
            path.append(self.nodes[k])
            k = self.predecessors[j, k]
        self._cached_shortest_paths[(i, j)] = path
        return path

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
        for node1 in self.nodes:
            for node2 in self.nodes:
                try:
                    node1_to_node2 = self.get_transform(node1, node2)
                    node2_to_node1 = self.get_transform(node2, node1)
                    node1_to_node2_inv = invert_transform(node2_to_node1)
                    consistent = consistent and np.allclose(node1_to_node2,
                                                            node1_to_node2_inv)
                except KeyError:
                    pass  # Frames are not connected
        return consistent
