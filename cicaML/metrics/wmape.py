import numpy as np


def wmape_v1(true, pred):
    # we take two series and calculate an output a wmape from it, not to be used in a grouping function

    # make a series called mape
    se_mape = abs(true - pred) / true

    # get a float of the sum of the true
    ft_actual_sum = true.sum()

    # get a series of the multiple of the true & the mape
    se_actual_prod_mape = true * se_mape

    # summate the prod of the true and the mape
    ft_actual_prod_mape_sum = se_actual_prod_mape.sum()

    # float: wmape of pred
    ft_wmape_forecast = ft_actual_prod_mape_sum / ft_actual_sum

    # return a float
    return ft_wmape_forecast


def wape(true, pred):
    return abs(true - pred).sum() / true.sum()


def wmape(true, pred, weights=None):
    if callable(weights):
        weights = weights(true, pred)

    if weights is None:
        weights = np.ones_like(true)

    return (abs(true - pred) * weights).sum() / (true * weights).sum()
