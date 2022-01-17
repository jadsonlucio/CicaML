import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

from cicaML.utils.functools import identity
from cicaML.utils.array import flatten


def window(array, step, stride):
    """
    Creates a rolling window.
    :param array: The array.
    :param step: The step size.
    :param stride: The stride size.
    :return: The rolling window.
    """
    groups = []
    for x in range(0, len(array) - stride + 1, step):
        groups.append(array[x : x + stride])

    return groups


def create_x_y(
    data,
    x_size,
    y_size,
    step_size=1,
    dataY=None,
    x_out_func=identity,
    y_out_func=identity,
):
    """
    Creates x and y data for a given data set.
    :param data: The data set.
    :param step: The step size.
    :param stride_x: The stride size for the x data.
    :param stride_y: The stride size for the y data.
    :param aggregation_func: The aggregation function.
    :return: x and y data.
    """

    if dataY is None:
        dataY = data

    x = window(data[:-y_size], step_size, x_size)
    y = window(dataY[x_size:], step_size, y_size)

    min_len = min(len(x), len(y))

    x = list(map(x_out_func, x[:min_len]))
    y = list(map(y_out_func, y[:min_len]))

    return np.array(x), np.array(y)


def train_test_split(
    input_data,
    output_data,
    input_size,
    output_size,
    train_size=0.8,
    step_size=1,
    x_out_func=identity,
    y_out_func=sum,
):
    if isinstance(train_size, float) and train_size > 1.0:
        raise ValueError("train_size must be a float between 0 and 1")

    x, y = create_x_y(
        input_data,
        input_size,
        output_size,
        dataY=output_data,
        step_size=step_size,
        x_out_func=x_out_func,
        y_out_func=y_out_func,
    )
    x = list(map(flatten, x))
    y = list(map(flatten, y))
    trainX, testX, trainY, testY = sk_train_test_split(
        x, y, train_size=train_size, shuffle=False
    )

    return trainX, testX, trainY, testY
