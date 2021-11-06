import unittest
import pandas as pd
from vale_ia.tsa.time_serie import TimeSeries


class TestTimeSerie(unittest.TestCase):
    def setUp(self):
        index = pd.date_range(start='1/1/2018', periods=9)
        self.time_serie = TimeSeries(data=[-20, 1, 1, 2, 3, 4, 20, 5, 7],
                                     index=index)

    def test_get_outliers(self):
        index = self.time_serie.outliers(method='IQR')
        self.assertEqual(self.time_serie[index].values.tolist(), [-20, 20])
        index = self.time_serie.outliers(method='Z-SCORE', std=1.7)
        print(self.time_serie[index].values.tolist())
        self.assertEqual(self.time_serie[index].values.tolist(), [-20, 20])
