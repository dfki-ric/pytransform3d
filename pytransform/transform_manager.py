import numpy as np
from .transformations import invert_transform


class TransformManager(object):
    """TODO document me"""
    def __init__(self):
        self.transforms = {}
        self.tree = {}

    def add_transform(self, from_frame, to_frame, A2B):
        """TODO document me"""
        self.transforms[(from_frame, to_frame)] = A2B
        if from_frame not in self.tree:
            self.tree[from_frame] = []
        self.tree[from_frame].append(to_frame)  # TODO check if it already exists
        return self

    def get_transform(self, from_frame, to_frame):
        """TODO document me"""
        if (from_frame, to_frame) in self.transforms:
            return self.transforms[(from_frame, to_frame)]
        elif (to_frame, from_frame) in self.transforms:
            return invert_transform(self.transforms[(to_frame, from_frame)])
        else:
            raise KeyError("Cannot compute transform from frame '%s' to frame "
                           "'%s'." % (from_frame, to_frame))
