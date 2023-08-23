import numpy as np


def array_namespace(*args):
    try:
        import array_api_compat
        return array_api_compat.array_namespace(*args)
    except ImportError:
        return np
