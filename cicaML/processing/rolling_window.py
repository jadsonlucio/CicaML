from typing import Any, Callable, Union
import numpy as np
from sklearn.model_selection import train_test_split as sk_train_test_split

from cicaML.utils.functools import identity
from cicaML.utils.array import flatten
from cicaML.types.base import GenericList
from cicaML.types.ml import TrainData
from cicaML.processing.decorators import processing_method


@processing_method(name="window", input_type="list", output_type="list")
def window(array: GenericList, step: int, stride: int) -> list:
    """
    Returns a sliding window of the given array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to create a sliding window from.
    step : int
        The step size of the sliding window.
    stride : int
        The size of the window.

    Returns
    -------
    numpy.ndarray
        The sliding window of the given array.

    Example
    -------
    >>> window([1, 2, 3, 4, 5, 6], 2, 2)
    [[1, 2], [3, 4], [5, 6]]

    """

    groups = []
    for x in range(0, len(array) - stride + 1, step):
        groups.append(array[x : x + stride])
    return groups


@processing_method(name="create_x_y", input_type="list", output_type="list")
def create_x_y(
    data: GenericList,
    x_size: int,
    y_size: int,
    step_size: int = 1,
    dataY: Union[GenericList, None] = None,
    x_out_func: Callable[[GenericList], Any] = identity,
    y_out_func: Callable[[GenericList], Any] = identity,
):
    """
    Creates a list of x and y values from the given data.

    Parameters
    ----------
    data : numpy.ndarray
        The data to create x values from.
    x_size : int
        The size of the x values.
    y_size : int
        The size of the y values.
    step_size : int
        The step size of the sliding window.
    dataY : numpy.ndarray
        The data to create y values from.
    x_out_func : function
        The function to apply to the x values.
    y_out_func : function
        The function to apply to the y values.

    Returns
    -------
    list
        The list of x and y values.

    Example
    -------
    >>> create_x_y([1, 2, 3, 4, 5, 6], 2, 2)
    ([[1, 2], [2, 3], [3,4]], [[3, 4], [4, 5], [5, 6]])
    >>> create_x_y([1, 2, 3, 4, 5, 6], 2, 2, step_size=2)
    ([[1, 2], [3, 4]], [[3, 4], [5, 6]])
    >>> create_x_y([1, 2, 3, 4, 5, 6], 2, 2, step_size=2, dataY=[7, 8, 9, 10, 11, 12])
    ([[1, 2], [3, 4]], [[9, 10], [11, 12]])
    >>> create_x_y([1, 2, 3, 4, 5, 6], 2, 2, step_size=2, dataY=[7, 8, 9, 10, 11, 12], x_out_func=sum)
    ([3, 7], [[9, 10], [11, 12]])
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
    input_data: GenericList,
    output_data: GenericList,
    input_size: int,
    output_size: int,
    train_size: float = 0.8,
    step_size: int = 1,
    x_out_func: Callable[[GenericList], Any] = identity,
    y_out_func: Callable[[GenericList], Any] = sum,
) -> TrainData:
    """
    Splits the given data into training and test data.

    Parameters
    ----------
    input_data : numpy.ndarray
        The input data to split.
    output_data : numpy.ndarray
        The output data to split.
    input_size : int
        The size of the input data.
    output_size : int
        The size of the output data.
    train_size : float
        The percentage of the data to use for training.
    step_size : int
        The step size of the sliding window.
    x_out_func : function
        The function to apply to the x values.
    y_out_func : function
        The function to apply to the y values.

    Returns
    -------
    tuple[list, list, list, list]
        The trainX, testX, trainY, testY values.

    Example
    -------
    >>> train_test_split([1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15], 2, 2)
    ([[1, 2], [2, 3], [3, 4]], [[4,5]], [23, 25, 27], [29])

    Raises
    -----
    ValueError
        If the train_size is not between 0 and 1.

    """
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
