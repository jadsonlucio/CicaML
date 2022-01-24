from .debug import debug_train
from .rolling_window import window
from .outlier_removal import std_outlier_removal
from .rolling_window import create_x_y, window, train_test_split  # noqa
from cicaML.utils.array import flatten

PROCESSING_METHODS = {
    "std_outlier_removal": std_outlier_removal,
    "window": window,
    "flatten": lambda x: list(map(flatten, x)),
    "df_train_test_split": lambda df, **params: train_test_split(
        df[params.pop("input_cols")].values, df[params.pop("output_cols")].values, **params
    ),
    "test": lambda x: x,
    "debug_train": debug_train
}
