from cicaML.ml.model import Model
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


class SklearnMLPRegressor(Model):
    """
    Sklearn regressor model.

    Parameters
    ----------
    hidden_layer_sizes : list
        The hidden layer sizes.

    """

    register_name = "sklearn_mlp_regressor"
    version = "1.0"

    def __init__(self, hidden_layer_sizes=(100, 100, 100), model=None, *args, **kwargs) -> None:
        if model is not None:
            self._model = model
        else:
            self._model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, random_state=42
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


class SklearnRandomForestRegressor(Model):
    """
    Random forest model.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest.
    random_state : int
        Random seed.
    """
    register_name = "sklearn_random_forest_regressor"
    version = "1.0"

    def __init__(self, n_estimators=10, random_state=42, model=None, *args, **kwargs) -> None:
        if model is not None:
            self._model = model
        else:
            self._model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

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
