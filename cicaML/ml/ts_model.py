from cicaML.tsa.time_serie import TimeSeries
from cicaML.ml.model import Model
from cicaML.utils.functools import identity
from cicaML.utils.array import rolling_window


class TSModel(Model):
    def __init__(
        self,
        x_size,
        model_obj,
        name,
        version,
        data_origin,
        y_size=1,
        step_size=1,
        x_out_func=identity,
        y_out_func=identity,
        evaluation_func=None,
        hyperparameters=None,
        metadata=None,
        fitted=False,
        evaluation_score=None,
    ):
        super().__init__(
            model_obj=model_obj,
            name=name,
            version=version,
            data_origin=data_origin,
            evaluation_func=evaluation_func,
            hyperparameters=hyperparameters,
            metadata=metadata,
            fitted=fitted,
            evaluation_score=evaluation_score,
        )

        self.x_size = x_size
        self.y_size = y_size
        self.step_size = step_size
        self.x_out_func = x_out_func
        self.y_out_func = y_out_func

    @property
    def model_params(self):
        params = super().model_params
        params["x_size"] = self.x_size
        params["y_size"] = self.y_size
        params["step_size"] = self.step_size
        params["x_out_func"] = self.x_out_func
        params["y_out_func"] = self.y_out_func

        return params

    def train_serie(self, array, train_size=0.75):
        ts = TimeSeries(data=array)
        trainX, testX, trainY, testY = ts.create_ts_train_test(
            x_size=self.x_size,
            y_size=self.y_size,
            train_size=train_size,
            step_size=self.step_size,
            x_out_func=self.x_out_func,
            y_out_func=self.y_out_func,
        )

        return self.fit(trainX, testX, trainY, testY)

    def create_X_y(self, array):
        X = rolling_window(
            array[: -self.y_size], self.x_size, self.step_size, func=self.x_out_func
        )
        y = rolling_window(
            array[self.x_size :], self.y_size, self.step_size, func=self.y_out_func
        )

        return X, y
