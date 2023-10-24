from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Collection, Tuple

import pandas as pd
from matplotlib.axes import Axes

from experiments.uci.metadata import BenchmarkMetadata
from benchmark.registry import Registry

# Avoid cyclic imports:
if TYPE_CHECKING:
    from grouping import GroupingSpec
else:
    GroupingSpec = Any


class Plotter(ABC):
    """
    Strategy for how to create a plot.
    """

    name: str
    """
    Name of this plotter.
    """

    @abstractmethod
    def plot(
        self,
        ax: Axes,
        file_key: Tuple[str, ...],
        column_key: Tuple[str, ...],
        row_key: Tuple[str, ...],
        line_by: GroupingSpec,
        metrics_df: pd.DataFrame,
        metadata: Collection[BenchmarkMetadata],
    ) -> None:
        """ Plot the given data to the given `ax`. """


PLOTTERS: Registry[Plotter] = Registry()


PlotterFn = Callable[
    [
        Axes,
        Tuple[str, ...],  # file_key
        Tuple[str, ...],  # column_key
        Tuple[str, ...],  # row_key
        GroupingSpec,  # line_by
        pd.DataFrame,  # metrics_df
        Collection[BenchmarkMetadata],
    ],
    None,
]
"""
A function that can be used as a :class:`Plotter`.

Plot the given data to the given `Axes`.
"""


class FnPlotter(Plotter):
    """
    Adapter from a function to a :class:`Plotter`.
    """

    def __init__(self, name: str, fn: PlotterFn) -> None:
        self.name = name
        self._fn = fn

    def plot(
        self,
        ax: Axes,
        file_key: Tuple[str, ...],
        column_key: Tuple[str, ...],
        row_key: Tuple[str, ...],
        line_by: GroupingSpec,
        metrics_df: pd.DataFrame,
        metadata: Collection[BenchmarkMetadata],
    ) -> None:
        self._fn(ax, file_key, column_key, row_key, line_by, metrics_df, metadata)


def make_plotter(fn: PlotterFn) -> FnPlotter:
    """
    Decorator for turning a function into a :class:`Plotter`.
    """
    name = fn.__name__
    result = FnPlotter(name, fn)
    PLOTTERS.add(result)
    return result