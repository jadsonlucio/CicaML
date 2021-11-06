import pandas as pd
from .df_plot import DFPlotEngine, SeriePlotEngine  # noqa


class CustomSerie(pd.Series):
    def __init__(self, *args, **kwargs) -> None:
        plot_engine_cls = kwargs.pop("plot_engine_cls", None)
        super().__init__(*args, **kwargs)

        if plot_engine_cls is None:
            plot_engine_cls = DFPlotEngine

        self.plot_engine = plot_engine_cls(self)

    @property
    def px_plot(self):
        return self.plot_engine


class CustomDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        plot_engine_cls = kwargs.pop("plot_engine_cls", None)
        super().__init__(*args, **kwargs)

        if plot_engine_cls is None:
            plot_engine_cls = DFPlotEngine

        self.plot_engine = plot_engine_cls(self)

    @property
    def px_plot(self):
        return self.plot_engine

    @property
    def _constructor(self):
        return CustomDataFrame

    @property
    def _constructor_sliced(self):
        return CustomSerie
