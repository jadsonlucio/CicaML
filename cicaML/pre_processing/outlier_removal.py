import numpy as np


def std_outlier_removal(data, standard_deviations=3, filling_mode="mean"):
    """
    Removes outliers from a dataset based on the standard deviation.

    Parameters
    ----------
    data : numpy.ndarray
        The dataset to remove outliers from.
    standard_deviations : int
        The number of standard deviations to remove.
    filling_mode : str
        The mode to fill the outliers with.

    Returns
    -------
    numpy.ndarray
        The dataset without outliers.
    """
    mean = data.mean()
    std = data.std()
    outlier_index = np.abs(data - mean) > standard_deviations * std

    if filling_mode == "mean":
        mean_without_outliers = data[~outlier_index].mean()
        data[outlier_index] = mean_without_outliers
    else:
        raise Exception(f"filling mode {filling_mode} note supported")

    return data
