import numpy as np
from sklearn.preprocessing import OneHotEncoder
from cicaML.processing.decorators import processing_method
from .rolling_window import train_test_split


@processing_method(name="weekday", input_type="df", output_type="column")
def weekday(df):
    """
    Returns the weekday of the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the weekday from.

    Returns
    -------
    array
        The weekday of the given dataframe.

    """
    weekdays = df.index.day_name()[:7]
    enc = OneHotEncoder(handle_unknown="ignore")

    df_weekday = np.array(df.index.weekday)
    df_weekday_enc = enc.fit_transform(df_weekday.reshape(-1, 1)).toarray().astype(int)

    return df_weekday_enc


@processing_method(name="month", input_type="df", output_type="column")
def month(df):
    """
    Returns the month of the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to get the month from.

    Returns
    -------
    array
        The month of the given dataframe.
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    df_month = np.array(df.index.month)
    df_month_enc = enc.fit_transform(df_month.reshape(-1, 1)).toarray().astype(int)

    return df_month_enc


@processing_method(name="df_train_test_split", input_type="df", output_type="column")
def df_train_test_split(df, input_cols, output_cols, **kwargs):
    """
    Splits a dataframe into train and test data.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to split.
    input_cols : list
        The input columns to use.
    output_cols : list
        The output columns to use.
    **kwargs
        The keyword arguments to pass to the train_test_split function.

    """

    return train_test_split(
        df[input_cols].values, df[output_cols].values, **kwargs
    )
