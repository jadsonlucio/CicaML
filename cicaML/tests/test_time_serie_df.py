import unittest
import pandas as pd
from cicaML.tsa.time_serie import TimeSeriesDF


class TestTimeSerie(unittest.TestCase):
    def setUp(self):
        index = pd.date_range(start="1/1/2018", periods=9)
        self.time_serie_df = TimeSeriesDF(
            data={"test": [-20, 1, 1, 2, 3, 4, 20, 5, 7]}, index=index
        )

    def test_get_outliers(self):
        time_serie_df = self.time_serie_df.filter_outliers("test", "IQR")
        filter_df = self.time_serie_df[
            [False, True, True, True, True, True, False, True, True]
        ]
        assert time_serie_df.equals(filter_df)
