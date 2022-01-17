from sklearn.metrics import r2_score, mean_squared_error
from .wmape import wmape

EVALUATION_METRICS = {
    'r2_score': r2_score,
    'mean_squared_error': mean_squared_error,
    "wmape": wmape,
}
