import joblib
import logging
import itertools

from sklearn.metrics import r2_score, mean_squared_error
from cicaML.ml.model_result import ModelResult

logger = logging.getLogger(__name__)


def validate_model(func):
    def wrapper(obj, *args, **kwargs):
        if not obj.fitted:
            raise Exception("Model not fitted")

        return func(obj, *args, **kwargs)

    return wrapper


evaluation_funcs = {"r2_score": r2_score, "mse": mean_squared_error}


class Model:
    def __init__(
        self,
        model_obj,
        name,
        version,
        data_origin,
        evaluation_func=None,
        hyperparameters=None,
        metadata=None,
        fitted=False,
        evaluation_score=None,
    ):
        self.name = name
        self.version = version
        self.data_origin = data_origin
        self.hyperparameters = hyperparameters or {}
        self.metadata = metadata or {}
        self.evaluation_score = evaluation_score
        self._model = model_obj
        self.fitted = fitted
        self.evaluation_func_name = None
        if isinstance(evaluation_func, str):
            self.evaluation_func_name = evaluation_func
            if evaluation_func not in evaluation_funcs:
                raise Exception(f"evaluation function {evaluation_func} not found")
            evaluation_func = evaluation_funcs[evaluation_func]

        self.evaluation_func = evaluation_func

    @property
    def model_params(self):
        return {
            "name": self.name,
            "version": self.version,
            "data_origin": self.data_origin,
            "hyperparameters": self.hyperparameters,
            "metadata": self.metadata,
            "evaluation_score": self.evaluation_score,
            "evaluation_func": self.evaluation_func_name,
            "fitted": self.fitted,
            "model_obj": self._model,
        }

    @validate_model
    def save(self, output):
        joblib.dump(self.model_params, output)

    def fit(self, x_train, x_test, y_train, y_test):
        self._model.fit(x_train, y_train)
        if x_test and y_test:
            self.evaluation_score = self.evaluation_func(
                y_test, self._model.predict(x_test)
            )
        self.fitted = True

        return ModelResult(self, x_train, x_test, y_train, y_test)

    @validate_model
    def predict(self, X):
        return self._model.predict(X)

    @validate_model
    def evaluate(self, x_data, evaluation_func=None):
        evaluation_func = evaluation_func or self.evaluation_func
        if evaluation_func is None:
            raise Exception("evaluation_func param must be passed")

        evaluation_func = evaluation_func or self.evaluation_func

    @classmethod
    def load(cls, target):
        model_params = joblib.load(target)
        return cls(**model_params)

    @classmethod
    def grid_search(
        cls,
        estimator,
        param_grid,
        x_train,
        x_test,
        y_train,
        y_test,
        name,
        version,
        data_origin,
        evaluation_func,
        metadata=None,
        *args,
        **kwargs,
    ):
        result = []
        for element in itertools.product(*param_grid.values()):
            hyperparameters = dict(zip(param_grid.keys(), element))
            model_obj = estimator(**hyperparameters)
            model = cls(
                model_obj,
                name=name,
                version=version,
                data_origin=data_origin,
                evaluation_func=evaluation_func,
                metadata=metadata,
                hyperparameters=hyperparameters,
                *args,
                **kwargs,
            )
            model.fit(x_train, x_test, y_train, y_test)
            result.append((model.evaluation_score, model))
            logger.info(
                f"model {name} with hyperparams {hyperparameters} "
                f"has score {model.evaluation_score}"
            )

        result = sorted(result, key=lambda a: a[0])
        return result
