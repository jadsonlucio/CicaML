from typing import Union

import joblib
import logging

from abc import ABC, abstractclassmethod

from pandas import DataFrame
from cicaML.data_manager.manager import DataManager
from cicaML.ml.model_result import ModelResult
from cicaML.metrics import EVALUATION_METRICS

logger = logging.getLogger(__name__)


def validate_model(func):
    def wrapper(obj, *args, **kwargs):
        if not obj.fitted:
            raise Exception("Model not fitted")

        return func(obj, *args, **kwargs)

    return wrapper


class Model(ABC):
    register_name = None
    version = None

    subclasses = {}

    def __init__(
        self,
        name: str,
        version: str,
        data_origin: str,
        evaluation_func=EVALUATION_METRICS["wmape"],
        hyperparameters=None,
        metadata=None,
        fitted=False,
        score=None,
        data_manager: Union[DataManager, dict, None] = None,
    ):
        if isinstance(data_manager, dict):
            data_manager = DataManager(
                data_manager["processing_methods"], data_manager["variables"]
            )
        data_manager = data_manager or DataManager([], {})
        self.name = name
        self.version = version
        self.data_origin = data_origin
        self.hyperparameters = hyperparameters or {}
        self.metadata = metadata or {}
        self.score = score
        self.fitted = fitted
        self.evaluation_func_name = None
        self.data_manager = data_manager

        if isinstance(evaluation_func, str):
            evaluation_func = evaluation_func.lower()
            if evaluation_func not in EVALUATION_METRICS:
                raise Exception(f"evaluation function {evaluation_func} not found")
            evaluation_func = EVALUATION_METRICS[evaluation_func]

        self.evaluation_func = evaluation_func

    @property
    def extra_params(self):
        return {}

    @property
    def build_params(self):
        return {
            "model_params": self.extra_params,
            "data_manager": self.data_manager.json,
            "identity_params": {
                "name": self.name,
                "version": self.version,
                "data_origin": self.data_origin,
                "metadata": self.metadata,
            },
            "train_result_params": {
                "evaluation_func": self.evaluation_func_name,
                "fitted": self.fitted,
                "score": self.score,
            },
            "register_name": self.register_name,
        }

    @property
    def results(self):
        return {
            "score": self.score,
        }

    @validate_model
    def save(self, output):
        joblib.dump(self.build_params, output)

    @validate_model
    def predict_raw(self, data: DataFrame):
        x = self.data_manager.get_variable("x", data)
        return self.predict(x)

    def fit_raw(self, data):
        trainX, testX, trainY, testY = self.data_manager.get_variable("train", data)
        return self.fit(trainX, testX, trainY, testY)

    @abstractclassmethod
    def _fit(self, x_train, y_train):
        pass

    def fit(self, x_train, x_test, y_train, y_test):
        self._fit(x_train, y_train)
        if x_test is not None and y_test is not None and self.evaluation_func:
            self.score = self.evaluation_func(y_test, self.predict(x_test))
        self.fitted = True

        return ModelResult(self, x_train, x_test, y_train, y_test)

    @abstractclassmethod
    @validate_model
    def predict(self, X):
        pass

    @staticmethod
    def load(target):
        build_params = joblib.load(target)
        class_ = Model.subclasses[build_params["register_name"]]
        return class_(
            data_manager=build_params["data_manager"],
            **build_params["model_params"],
            **build_params["identity_params"],
            **build_params["train_result_params"],
        )

    def __init_subclass__(cls, **kwargs):
        cls.subclasses[cls.register_name] = cls
        super().__init_subclass__(**kwargs)
