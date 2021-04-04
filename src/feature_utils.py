#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd


def df_from_cartesian_product(dict_in):
    """Generate a Pandas dataframe from Cartesian product of lists.

    Args:
        dict_in (Dictionary): Dictionary containing multiple lists,
        e.g. {"fea1": list1, "fea2": list2}

    Returns:
        df (Dataframe): Dataframe corresponding to the Caresian product of the
        lists
    """
    from itertools import product

    cart = list(product(*dict_in.values()))
    df = pd.DataFrame(cart, columns=dict_in.keys())
    return df


def lagged_features(df, lags):
    """Create lagged features based on time series data.

    Args:
        df (Dataframe): Input time series data sorted by time
        lags (List): Lag lengths

    Returns:
        fea (Dataframe): Lagged features
    """
    df_list = []
    for lag in lags:
        df_shifted = df.shift(lag)
        df_cols = df_shifted.columns
        df_shifted.columns = [x + "_lag" + str(lag) for x in df_cols]
        df_list.append(df_shifted)
    fea = pd.concat(df_list, axis=1)
    return fea


def moving_averages(df, start_step, window_size=None):
    """Compute averages of every feature over moving time windows.

    Args:
        df (Dataframe): Input features as a dataframe
        start_step (Integer): Starting time step of rolling mean
        window_size (Integer): Windows size of rolling mean

    Returns:
        fea (Dataframe): Dataframe consisting of the moving averages
    """
    if window_size is None:
        # Use a large window to compute average over all historical data
        window_size = df.shape[0]
    fea = (
        df.shift(start_step)
        .rolling(min_periods=1, center=False, window=window_size)
        .mean()
    )
    fea.columns = fea.columns + "_mean"
    return fea


def combine_features(df, lag_fea, lags, window_size, used_columns):
    """Combine lag features, moving average features, and orignal features in
    the data.

    Args:
        df (Dataframe): Time series data including the target series and
        external features
        lag_fea (List): A list of column names for creating lagged features
        lags (Numpy Array): Numpy array including all the lags
        window_size (Integer): Window size of rolling mean
        used_columns (List): A list containing the names of columns that are
        needed in the input dataframe (including the target column)

    Returns:
        fea_all (Dataframe): Dataframe including all the features
    """
    lagged_fea = lagged_features(df[lag_fea], lags)
    moving_avg = moving_averages(df[lag_fea], min(lags), window_size)
    fea_all = pd.concat([df[used_columns], lagged_fea, moving_avg], axis=1)
    return fea_all
