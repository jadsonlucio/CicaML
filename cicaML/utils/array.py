import json
import numpy as np
from cicaML.utils.functools import identity


def is_array(array):
    if isinstance(array, list) or isinstance(array, tuple) or isinstance(array, np.ndarray):
        return True

    return False


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


def flatten(array):
    if not is_array(array):
        return array

    flatted_list = []
    for arr in array:
        flatten_arr = flatten(arr)
        if is_array(flatten_arr):
            flatted_list.extend(flatten_arr)
        else:
            flatted_list.append(flatten_arr)

    return np.array(flatted_list)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
