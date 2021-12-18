def fill_nan_tsa(tsa, method='mean'):
    """
    Fill NaN values in a time series array.

    Parameters
    ----------
    tsa : numpy.ndarray
        Time series array.
    method : str, optional
        Method to fill NaN values.
        Default is 'mean'.

    Returns
    -------
    numpy.ndarray
        Time series array with NaN values filled.

    """
    if method == 'mean':
        return tsa.fillna(tsa.mean())
    elif method == 'median':
        return tsa.fillna(tsa.median())
    elif method == 'zero':
        return tsa.fillna(0)
    else:
        raise ValueError('Method not recognized.')
