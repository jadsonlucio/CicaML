# https://pandas.pydata.org/pandas-docs/stable/development/extending.html

import numpy as np
import pandas as pd
import nolds
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
from cicaML.utils.functools import identity
from cicaML.tsa.time_serie_plot import TimeSeriePlotEngine
from cicaML.utils.pandas.df import CustomSerie, CustomDataFrame
from cicaML.pre_processing.rolling_window import create_x_y


from cicaML.utils.collections import merge
from cicaML.utils.plotly_utils import hline

group_methods = {
    "Dia": lambda time_serie: time_serie.resample("D").sum(),
    "Média semanal": lambda time_serie: time_serie.get_trend("moving_average", freq=7),
}

week_days = ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"]
months_year = [
    "jan",
    "fev",
    "mar",
    "abr",
    "mai",
    "jun",
    "jul",
    "ago",
    "set",
    "out",
    "nov",
    "dez",
]


def ts_filter_dec(name, convert_kwargs=None, *args, **kwargs):
    kwargs_attrs = [
        "start_date",
        "end_date",
        "group",
        "remove_outliers",
        "replace_outliers",
    ]

    convert_kwargs = convert_kwargs or {}

    def wrapper(func):
        def wrapper_2(*args, **kwargs):
            convert_kwargs_ = {}
            for attr in kwargs_attrs:
                if attr in kwargs:
                    convert_kwargs_[attr] = kwargs.pop(attr)

            time_serie = func(*args, **kwargs)
            time_serie.name = name
            convert_kwargs_ = merge(convert_kwargs_, convert_kwargs)

            return time_serie.convert(**convert_kwargs_)

        return wrapper_2

    return wrapper


class TimeSeries(CustomSerie):
    def __init__(self, *args, **kwargs):
        super().__init__(plot_engine_cls=TimeSeriePlotEngine, *args, **kwargs)

    def create_train_test(
        self,
        x_size,
        y_size,
        train_size=0.75,
        step_size=1,
        x_out_func=identity,
        y_out_func=identity,
    ):
        if isinstance(train_size, float) and train_size > 1.0:
            raise ValueError("train_size must be a float between 0 and 1")

        x, y = create_x_y(self, x_size, y_size,step_size=step_size, x_out_func=x_out_func, y_out_func=y_out_func)
        trainX, testX, trainY, testY = train_test_split(x, y, train_size=train_size, shuffle=False)

        return trainX, testX, trainY, testY

    def seasonal_decompose(self, freq, model="aditive"):
        d = seasonal_decompose(self.values, model=model, freq=freq)
        trend = TimeSeries(d.trend, self.index)
        seasonality = TimeSeries(d.seasonal, self.index)
        noise = TimeSeries(d.resid, self.index)

        return TimeSeriesDF(
            data={"trend": trend, "seasonality": seasonality, "noise": noise}
        )

    def filter_by_date_range(self, start_date=None, end_date=None):
        serie = self
        if start_date is not None:
            serie = serie[serie.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            serie = serie[serie.index <= pd.to_datetime(end_date)]

        return serie

    def convert(
        self,
        group=None,
        start_date=None,
        end_date=None,
        remove_outliers=None,
        replace_outliers=None,
    ):
        serie = self.filter_by_date_range(start_date=start_date, end_date=end_date)

        if remove_outliers:
            serie = serie.remove_outliers(**remove_outliers)
        elif replace_outliers:
            serie = serie.replace_outliers(**replace_outliers)

        if group:
            freq = group.pop("freq", None)
            agg_func = group.pop("agg_func")
            serie = serie.groupby(pd.Grouper(level=0, freq=freq)).agg(agg_func)

        return serie

    def format_money(self, currency="BR"):
        if currency == "BR":
            return self.map("R$ {:,.2f}".format)

    def daily_serie(self, agg_func=sum):
        ts = self.groupby(pd.Grouper(level=0, freq="D")).agg(agg_func)
        ts = ts.rename(f"{self.name} - Diário")

        return ts

    def week_serie(self, agg_func=sum):
        ts = self.groupby(pd.Grouper(level=0, freq="W")).agg(agg_func)
        ts = ts.rename(f"{self.name} - Semanal")

        return ts

    def weekday_serie(self, agg_func=sum):
        week_days = {i: dia for i, dia in enumerate(week_days)}

        weekday_serie = self.groupby(self.index.weekday).agg(agg_func)
        weekday_serie.index = map(lambda i: week_days[i], weekday_serie.index)
        weekday_serie = weekday_serie.rename(f"{self.name} - Dia semana")

        return weekday_serie

    def month_serie(self, agg_func=sum):
        months_year = {i: mes for i, mes in enumerate(months_year, start=1)}

        if len(self) == 0:
            return pd.Series(
                index=list(months_year.values()), data=[0 for i in months_year]
            )

        month_serie = self.groupby(self.index.month).agg(agg_func)
        month_serie.index = map(lambda i: months_year[i], month_serie.index)
        month_serie = month_serie.rename(f"{self.name} - Mensal")

        return month_serie

    # regression methods

    def lr_serie(self):
        dataX = np.array([i for i in range(1, len(self) + 1)]).reshape(-1, 1)

        dataY = [0 if np.isnan(v) else v for v in self.values]
        linear_model = LinearRegression()
        linear_model.fit(dataX, dataY)

        pred = linear_model.predict(dataX)
        ts = TimeSeries(data=pred, index=self.index)
        ts = ts.rename(f"Tendência {self.name}")
        ts.linear_model = linear_model

        return ts

    def trend_rate(self, method="linear_regression"):
        if method == "linear_regression":
            ts = self.lr_serie()
            return (ts[-1] - ts[0]) / abs(ts[0])

    # anomaly detection methods

    def outliers_bounds(self, method="IQR", **kwargs):
        if method == "IQR":
            Q1 = self.quantile(kwargs.get("Q1", 0.25))
            Q3 = self.quantile(kwargs.get("Q3", 0.75))
            IQR = Q3 - Q1
            k1, k2 = kwargs.get("k1", 1.5), kwargs.get("k2", 1.5)
            min_range = Q1 - k1 * IQR
            max_range = Q3 + k2 * IQR

        return min_range, max_range

    def outliers(self, method="IQR", **kwargs):
        if method == "IQR":
            min_range, max_range = self.outliers_bounds(method, **kwargs)
            return (self.values < min_range) | (self.values > max_range)
        if method == "Z-SCORE":
            if len(self.values) == 0 or self.values.std() == 0:
                return np.zeros(len(self.values), dtype=bool)

            return ~(
                abs((self.values - self.values.mean()) / self.values.std())
                < kwargs.get("std", 3)
            )
        if method.startswith("GROUP"):
            group = kwargs.pop("group")
            if group is None:
                raise ValueError("group must be informed")

            method = method.split("_")[1]

            return self.groupby(group).transform(
                lambda serie: serie.outliers(method=method, **kwargs)
            )

    def replace_outliers(
        self, method="IQR", fill_func=lambda x, idx: np.mean(x), **kwargs
    ):
        serie_copy = self.copy()
        outliers_idxs = self.outliers(method, **kwargs)

        if method.startswith("GROUP"):
            group = kwargs.pop("group")
            if group is None:
                raise ValueError("group must be informed")

            method = method.split("_")[1]

            serie_copy = self.groupby(group).transform(
                lambda serie: serie.replace_outliers(
                    method=method, fill_func=fill_func, **kwargs
                )
            )

            return serie_copy

        serie_no_outliers = self[~outliers_idxs]
        serie_copy[outliers_idxs] = [
            fill_func(serie_no_outliers, idx) for idx in outliers_idxs
        ]

        return serie_copy

    def remove_outliers(self, method="IQR", **kwargs):
        outliers_idxs = self.outliers(method, **kwargs)
        serie_no_outliers = self[~outliers_idxs]

        return serie_no_outliers

    def get_trend(self, method="moving_average", **kwargs):
        if method == "moving_average":
            df = self.seasonal_decompose(freq=kwargs.get("freq"))
            return df["trend"]

    # plot methods

    def plot_box(
        self,
        fig=None,
        create_fig=False,
        name=None,
        trace_kwargs=None,
        split_outliers=None,
        *args,
        **kwargs,
    ):
        name = name or self.name
        trace_kwargs = trace_kwargs or {}
        if fig is None and create_fig is True:
            fig = go.Figure()

        ts = self
        objs = []

        if split_outliers is not None:
            outliers = self.outliers(**split_outliers)
            ts = self[~outliers]
            ts_outliers = self[outliers]

            if "x" in kwargs:
                outliers_labels = np.array(kwargs["x"])[outliers]
                kwargs["x"] = np.array(kwargs["x"])[~outliers]
            else:
                outliers_labels = [name] * len(outliers)

            scatter = go.Scatter(
                x=outliers_labels,
                y=ts_outliers.values,
                mode="markers",
                text=ts_outliers.index.strftime("%d/%m/%Y"),
                name="Dias fora da curva",
                visible="legendonly",
            )

            objs.append(scatter)

        box = go.Box(y=ts.values, name=name, *args, **kwargs)
        objs.append(box)

        if fig is not None:
            for obj in objs:
                fig.add_trace(obj, **trace_kwargs)

            return fig

        return tuple(objs)

    def plot_distribution(self, fig=None, name="", layout_kwargs=None, *args, **kwargs):
        pass

    def plot_line_outliers(
        self,
        fig,
        method="IQR",
        show_min=True,
        show_max=True,
        max_line_kwargs=None,
        min_line_kwargs=None,
        **kwargs,
    ):
        min_value, max_value = self.outliers_bounds(method=method, **kwargs)
        default_line_kwargs = {"color": "red", "width": 1, "dash": "dash"}
        max_line_kwargs = max_line_kwargs or default_line_kwargs
        min_line_kwargs = min_line_kwargs or default_line_kwargs

        if show_min is True:
            hline(
                fig,
                min_value,
                self.index[0],
                self.index[-1],
                min_line_kwargs,
                name="Limite Inferior",
            )
        if show_max is True:
            hline(
                fig,
                max_value,
                self.index[0],
                self.index[-1],
                max_line_kwargs,
                name="Limite Superior",
            )

        return fig

    # statics methods

    def entropy(self, method="sampen", **kwargs):
        if method == "sampen":
            return nolds.sampen(self, **kwargs)
        if method == "seasonal_decompose":
            df = self.seasonal_decompose()
            error_serie = df["noise"]
            return ((self - error_serie.abs()).abs() / self).abs().mean()

    def cdf(self, n_bins=50):
        """cumulative density function"""
        mu = self.mean()
        sigma = self.std()
        n_bins = 50

        # plot the cumulative histogram
        fig = plt.figure()
        n, bins, patches = plt.hist(
            self.values,
            n_bins,
            density=True,
            histtype="step",
            cumulative=True,
            label="Empirical",
        )
        plt.close(fig)

        # Add a line showing the expected distribution.
        y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -0.5 * (1 / sigma * (bins - mu)) ** 2
        )
        y = y.cumsum()
        y /= y[-1]

        return pd.Series(index=bins, data=y, name=f"Distribuição {self.name}")

    @property
    def day_name_br(self):
        return self.index.weekday.map(lambda day_idx: week_days[day_idx])

    @property
    def month_name_br(self):
        return self.index.month.map(lambda month_idx: months_year[month_idx])

    @property
    def min_date(self):
        return min(self.index)

    @property
    def max_date(self):
        return max(self.index)

    @staticmethod
    def from_df(df, col, date_col="datetime"):
        return TimeSeries(data=df[col].values, index=df[date_col].values)

    @property
    def _constructor(self):
        return TimeSeries

    def fill_missing_dates(self, fill_value=0, start_date=None, end_date=None):
        if start_date is None:
            date = self.index[0]

            if isinstance(date, datetime):
                date = date.date()

        if end_date is None:
            date = self.index[0]

            if isinstance(date, datetime):
                date = date.date()

        idx = pd.date_range(start_date, end_date)
        return self.reindex(idx, fill_value=fill_value)

    # pandas methods
    def not_found_values(self, values, raise_exception=False):
        unique_values = set(self.values)
        not_found_values = []
        for v in values:
            if v not in unique_values:
                not_found_values.append(v)

        if raise_exception and not_found_values:
            raise Exception(
                f"can't find the following values {not_found_values} in serie {self.name}"
            )


class TimeSeriesDF(CustomDataFrame):
    def filter_outliers(self, column, method="IQR", **kwargs):
        serie = self[column]
        no_outliers = ~serie.outliers(method=method, **kwargs)

        return self[no_outliers]

    def filter_by_date_range(self, start_date=None, end_date=None):
        df = self
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def create_train_test(
        self,
        x_size,
        y_size,
        inputs_cols,
        output_cols,
        train_size=0.8,
        step_size=1,
        x_out_func=identity,
        y_out_func=identity,
    ):
        if isinstance(train_size, float) and train_size > 1.0:
            raise ValueError("train_size must be a float between 0 and 1")

        x, y = create_x_y(self[inputs_cols].values, x_size, y_size, dataY=self[output_cols].values, step_size=step_size, x_out_func=x_out_func, y_out_func=y_out_func)
        trainX, testX, trainY, testY = train_test_split(x, y, train_size=train_size, shuffle=False)

        return trainX, testX, trainY, testY

    @classmethod
    def from_df(cls, df, date_index_column=None):
        if date_index_column is not None:
            df = df.set_index(pd.DatetimeIndex(df[date_index_column]))

        return cls(df)

    @property
    def _constructor(self):
        return TimeSeriesDF

    @property
    def _constructor_sliced(self):
        return TimeSeries


class GroupTimeSeries(TimeSeries):
    def __init__(self, agg_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_func = agg_func

    @property
    def daily_serie(self):
        return super().daily_serie(agg_func=self.agg_func)

    @property
    def week_serie(self):
        return super().week_serie(agg_func=self.agg_func)

    @property
    def month_serie(self):
        return super().month_serie(agg_func=self.agg_func)

    @property
    def weekday_serie(self):
        return super().weekday_serie(agg_func=self.agg_func)
