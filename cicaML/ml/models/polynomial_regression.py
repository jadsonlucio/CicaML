from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


class PolynomialRegression:
    def __init__(self, degree=1):
        self._model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                                ('linear',
                                 LinearRegression(fit_intercept=False))])

    def fit(self, trainY, trainX=None, testX=None, testY=None):
        if trainX is None:
            trainX = [[i] for i in range(1, len(trainY) + 1)]
        self.trainX = trainX
        self.trainY = trainY
        self._model.fit(trainX, trainY)

    def predict(self, X):
        return self._model.predict(X)

    def intersect(self, value):
        """ return ceil((value - b + v)/a)"""
        model = self._model.named_steps['linear']
        a = model.coef_[1]
        b = model.coef_[0]
        v = a * self.trainX[-1][0] + b
        x = self.trainX[-1][0] + 1
        while a * x + b - v < value:
            x += 1

        return x
