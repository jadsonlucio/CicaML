from pandas.core.frame import DataFrame, Series
import plotly.graph_objects as go
from typing import Any, Union
from vale_ia.utils.plotly.px_utils import px_grid_plot, px_heatmap, px_stacked_heatmap


class MultiplePlots:
    def __init__(self, datasource: Union[DataFrame, Series], **kwargs) -> None:
        self.datasource = datasource
        self.stack = []

    def render(self):
        result = self.datasource
        for element in self.stack:
            if element["type"] == "call":
                result = result.__call__(
                    *element.get("args", []), **element.get("kwargs", {})
                )
            elif element["type"] == "getattr":
                result = result.__getattribute__(element["name"])

        return result

    def __call__(self, *args, **kwargs):
        self.stack.append({"type": "call", "args": args, "kwargs": kwargs})
        return self

    def __getattribute__(self, name: str) -> Any:
        self.stack.append({"type": "getattr", "name": name})
        return self


class SeriePlotEngine:
    def __init__(self, serie) -> None:
        self._serie = serie

    def go_bar(self, *args, **kwargs):
        return go.Bar(x=self.index, y=self.values, name=self.name, *args, **kwargs)

    def go_scatter(self, *args, **kwargs):
        return go.Scatter(x=self.index, y=self.values, name=self.name, *args, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._serie.plot(*args, **kwds)


class DFPlotEngine:
    def __init__(self, dataframe) -> None:
        self._dataframe = dataframe

    def plot(self, *args, **kwargs):
        return self._dataframe.plot(*args, **kwargs)

    def group(
        self,
        group_cols,
        trace_plot_func,
        extra_kwargs=None,
        col=None,
        row=None,
        col_label=None,
        row_label=None,
        *args,
        **kwargs,
    ):
        return px_grid_plot(
            self._dataframe,
            group_cols=group_cols,
            trace_plot_func=trace_plot_func,
            extra_kwargs=extra_kwargs,
            col=col,
            row=row,
            col_label=col_label,
            row_label=row_label,
            *args,
            **kwargs,
        )

    def heatmap(
        self,
        row,
        col,
        values,
        hover_text=None,
        annot_text=None,
        colorscale="Oryel",
        *args,
        **kwargs,
    ):
        return px_heatmap(
            self._dataframe,
            row=row,
            col=col,
            values=values,
            hover_text=hover_text,
            annot_text=annot_text,
            colorscale=colorscale,
            *args,
            **kwargs,
        )

    def stacked_heatmap(
        self,
        row,
        values=None,
        annot_text=None,
        hover_text=None,
        colorscale="blues",
        data_func=None,
        extra_kwargs=None,
        *args,
        **kwargs,
    ):
        return px_stacked_heatmap(
            self._dataframe,
            row=row,
            values=values,
            annot_text=annot_text,
            hover_text=hover_text,
            colorscale=colorscale,
            data_func=data_func,
            extra_kwargs=extra_kwargs,
            *args,
            **kwargs,
        )

    def multiples(self, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)
