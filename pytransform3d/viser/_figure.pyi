from typing import Any, Union
import viser
import numpy.typing as npt


class Figure:
    def __init__(self): ...
    def plot_transform(
        self,
        A2B: Union[None, npt.ArrayLike] = ...,
        s: float = ...,
        name: Union[str, None] = ...,
        strict_check: bool = ...,
    ) -> Any: ...
    def plot_box(
        self,
        size: npt.ArrayLike = ...,
        A2B: npt.ArrayLike = ...,
        c: Union[None, npt.ArrayLike] = ...,
    ) -> Any: ...

def figure() -> Figure: ...
