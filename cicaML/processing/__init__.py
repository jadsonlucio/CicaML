from .debug import debug_train_data, test
from .outlier_removal import std_outlier_removal
from .rolling_window import create_x_y, window, train_test_split
from .df import weekday, month, df_train_test_split
from .array import flatten_nd_to_2d


PROCESSING_METHODS = {
    "std_outlier_removal": std_outlier_removal,
    "weekday": weekday,
    "month": month,
    "df_train_test_split": df_train_test_split,
    "create_x_y": create_x_y,
    "window": window,
    "train_test_split": train_test_split,
    "debug_train_data": debug_train_data,
    "test": test,
    "flatten": flatten_nd_to_2d,
}
