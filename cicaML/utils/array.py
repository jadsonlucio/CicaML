import json
import numpy as np
from cicaML.utils.functools import identity


def rolling_window(array, window_size, step_size=1, func=identity):
    response = []
    for i in range(0, len(array) - window_size, step_size):
        response.append(func(array[i : i + window_size]))

    return response


def array2string(array):
    array = np.array(array)
    array[array == None] = np.nan  # noqa
    array = array.astype(str)
    array[array == "nan"] = ""

    return array


def fill_matrix(array):
    row_lengths = []
    array = [list(row) for row in array]
    for row in array:
        row_lengths.append(len(row))

    max_length = max(row_lengths)
    for row in array:
        while len(row) < max_length:
            row.append(None)

    return np.array(array)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
