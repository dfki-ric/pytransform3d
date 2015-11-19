import numpy as np
from .transformations import invert_transform


class TransformManager(object):
    """TODO document me"""
    def __init__(self):
        self.nodes = {}

    def add_transform(self, from_frame, to_frame, A2B):
        """TODO document me"""
        self.nodes[(from_frame, to_frame)] = A2B
        return self

    def get_transform(self, from_frame, to_frame):
        """TODO document me"""
        if (from_frame, to_frame) in self.nodes:
            return self.nodes[(from_frame, to_frame)]
        elif (to_frame, from_frame) in self.nodes:
            return invert_transform(self.nodes[(to_frame, from_frame)])
        else:
            raise KeyError("Cannot compute transform from frame '%s' to frame "
                           "'%s'." % (from_frame, to_frame))
