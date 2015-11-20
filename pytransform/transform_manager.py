import numpy as np
import scipy.sparse as sp
from .transformations import invert_transform, concat


class TransformManager(object):
    """TODO document me"""
    def __init__(self):
        self.transforms = {}
        self.nodes = []
        self.i = []
        self.j = []
        self.predecessors = np.zeros((0, 0))

    def add_transform(self, from_frame, to_frame, A2B):
        """TODO document me"""
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
        """TODO document me"""
        if (from_frame, to_frame) in self.transforms:
            return self.transforms[(from_frame, to_frame)]
        elif (to_frame, from_frame) in self.transforms:
            return invert_transform(self.transforms[(to_frame, from_frame)])
        else:
            if from_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % from_frame)
            if to_frame not in self.nodes:
                raise KeyError("Unknown frame '%s'" % to_frame)
            i = self.nodes.index(from_frame)
            j = self.nodes.index(to_frame)
            if not np.isfinite(self.dist[i, j]):
                raise KeyError("Cannot compute path from frame '%s' to "
                               "frame '%s'." % (from_frame, to_frame))
            k = i
            path = []
            while k != -9999:
                path.append(self.nodes[k])
                k = self.predecessors[j, k]

            A2B = np.eye(4)
            for from_f, to_f in zip(path[:-1], path[1:]):
                A2B = concat(A2B, self.get_transform(from_f, to_f))
            return A2B
