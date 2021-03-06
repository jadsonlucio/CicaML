import numpy as np
from cicaML.ml.model import Model


class MovingAverage(Model):
    register_name = "moving_average"
    version = "1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit(self, trainX, trainY):
        pass

    def predict(self, X):
        output = []
        for i in X:
            output.append(np.mean(i))
        return np.array(output)
