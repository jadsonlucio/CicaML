from cicaML.types.ml import TrainData
from cicaML.processing.decorators import processing_method


@processing_method(
    name="debug_train_data", input_type="TrainData", output_type="TrainData"
)
def debug_train_data(train_data: TrainData) -> TrainData:
    """Print the train data.

    Parameters
    ----------
    train_data : TrainData
        The train data to print.

    Returns
    -------
    TrainData
        The train data.
    """

    trainX, testX, trainY, testY = train_data
    print(trainX, testX, trainY, testY)
    return trainX, testX, trainY, testY


@processing_method(name="test", input_type="Any", output_type="Any")
def test(x):
    """Test function.

    Parameters
    ----------
    x : Any
        The x value.

    Returns
    -------
    Any
        The x value.
    """
    return x
