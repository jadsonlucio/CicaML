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


def create_x_y(data, step, stride_x, stride_y, dataY=None, aggregation_func_x = lambda x: x, aggregation_func_y=sum):
    """
    Creates x and y data for a given data set.
    :param data: The data set.
    :param step: The step size.
    :param stride_x: The stride size for the x data.
    :param stride_y: The stride size for the y data.
    :param aggregation_func: The aggregation function.
    :return: x and y data.
    """

    data_copy = data.copy()

    if dataY is None:
        dataY = data

    dataY = dataY.values.copy()
    x = window(data_copy[:-stride_y], step, stride_x)
    y = window(dataY[stride_x:], step, stride_y)

    min_len = min(len(x), len(y))

    x = list(map(aggregation_func_x, x[:min_len]))
    y = list(map(aggregation_func_y, y[:min_len]))

    return x, y
