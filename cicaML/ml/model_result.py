import numpy as np


class ModelResult:
    def __init__(
        self,
        base_model,
        trainX: np.ndarray,
        testX: np.ndarray,
        trainY: np.ndarray,
        testY: np.ndarray,
    ):
        self.base_model = base_model
        self.trainX = trainX
        self.testX = testX
        self.trainX = trainX
        self.testY = testY
        self.trainY = trainY

        self.train_pred = self.base_model.predict(trainX)
        self.test_pred = self.base_model.predict(testX)

    def plot_test_prediction(self):
        pass
