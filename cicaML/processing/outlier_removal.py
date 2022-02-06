import numpy as np
from cicaML.processing.decorators import processing_method


@processing_method(name="std_outlier_removal", input_type="array", output_type="array")
def std_outlier_removal(data: np.ndarray, standard_deviations: int = 2, filling_mode: str = "mean"):
    """
    Removes outliers from a data based on the standard deviation.

    Parameters
    ----------
    data : numpy.ndarray
        The data to remove outliers from.
    standard_deviations : int
        The number of standard deviations to remove.
    filling_mode : str
        The mode to fill the outliers with.

    Returns
    -------
    numpy.ndarray
        The data without outliers.

    Raises
    ------
    ValueError
        If the filling mode is not supported.

    Example
    -------
    >>> std_outlier_removal(np.array([1,2,3,20, 30, 1000, 10000]))
    array([ 1,  2,  3, 20, 30, 1000, 176])

    """
    mean = data.mean()
    std = data.std()
    cut_off = standard_deviations * std
    lower, upper = mean - cut_off, mean + cut_off

    outliers = (data < lower) | (data > upper)

    if filling_mode == "mean":
        mean_without_outliers = data[~outliers].mean()
        data[outliers] = mean_without_outliers
    else:
        raise ValueError(f"filling mode {filling_mode} note supported")

    return data
