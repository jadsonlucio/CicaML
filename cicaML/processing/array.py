from cicaML.utils.array import flatten
from cicaML.processing.decorators import processing_method

@processing_method(name="flatten_nd_to_2d", input_type="list", output_type="list")
def flatten_nd_to_2d(array):
    """
    Flattens a n-dimensional array to a 2-dimensional array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to flatten.

    Returns
    -------
    numpy.ndarray
        The flattened array.

    Example
    -------
    >>> flatten_nd_to_2d(np.array([[1, [2, 3]], [4, [5, 6]]]))
    array([[1, 2, 3], [4, 5, 6]])
    """
    return list(map(flatten, array))
