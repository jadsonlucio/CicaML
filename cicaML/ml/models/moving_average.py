import numpy as np


class MovingAverage:
    def __init__(self):
        pass

    def fit(self, trainX, testX, trainY, testY):
        pass

    def predict(self, X):
        output = []
        for i in X:
            output.append(np.mean(i))

        return output
