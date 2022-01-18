from .moving_average import MovingAverage
from .sklearn import SklearnMLPRegressor, SklearnRandomForestRegressor
from .xgb import XGBRegressor

MODELS = {
    MovingAverage.register_name: MovingAverage,
    SklearnMLPRegressor.register_name: SklearnMLPRegressor,
    SklearnRandomForestRegressor.register_name: SklearnRandomForestRegressor,
    XGBRegressor.register_name: XGBRegressor,
}
