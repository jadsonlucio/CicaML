import numpy as np
from cicaML.utils.functools import identity

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
        groups.append(array[x:x+stride])

    return groups


def create_x_y(data, x_size, y_size, step_size=1, dataY=None, x_out_func = identity, y_out_func=identity):
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
