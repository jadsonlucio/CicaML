class ModelResult:
    def __init__(self, base_model, trainX, testX, trainY, testY):
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
