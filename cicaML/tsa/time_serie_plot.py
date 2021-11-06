from vale_ia.utils.pandas.df_plot import SeriePlotEngine, DFPlotEngine  # noqa


class TimeSeriePlotEngine(SeriePlotEngine):
    def __init__(self, time_serie, *args, **kwargs) -> None:
        super().__init__(time_serie, *args, **kwargs)
