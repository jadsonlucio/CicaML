import xgboost as xgb
from cicaML.ml.model import Model


class XGBRegressor(Model):
    register_name = "xgb_regressor"
    version = "1.0"

    def __init__(
        self,
        model=None,
        objective="reg:squarederror",
        learning_rate=0.01,
        max_depth=5,
        n_estimators=190,
        *args,
        **kwargs
    ) -> None:
        if model is not None:
            self._model = model
        else:
            self._model = xgb.XGBRegressor(
                objective=objective,
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )

        super().__init__(*args, **kwargs)

    def _fit(self, trainX, trainY):
        self._model.fit(trainX, trainY)

    def predict(self, X):
        return self._model.predict(X)

    @property
    def extra_params(self):
        return {
            "model": self._model,
        }
