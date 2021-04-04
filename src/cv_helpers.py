#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Tuple

import numpy as np
import pandas as pd


def show_cv_dates(fold_num, train_df, test_df, time_taken):
    train_dates = train_df.index.get_level_values("date")
    gap_start = train_dates.max() + pd.DateOffset(days=1)
    test_dates = test_df.index.get_level_values("date")
    gap_end = test_dates.min() - pd.DateOffset(days=1)
    df = train_df.reset_index().append(test_df.reset_index())
    n = len(df)
    assert n == len(df.drop_duplicates())
    test_size = test_df.groupby(level="symbol").size().value_counts().index[0]
    train_split_size = (
        train_df.groupby(level="symbol").size().value_counts().index[0]
    )
    gap_size = len(pd.date_range(gap_start.date(), gap_end.date()))
    print(
        f"{fold_num}|"
        f"{train_dates.min().date()} - {train_dates.max().date()} "
        f"({train_split_size})|"
        f"{gap_start.date()} - {gap_end.date()} ({gap_size})|"
        f"{test_dates.min().date()} - {test_dates.max().date()} "
        f"({test_size})|{time_taken:.2f}s"
    )


class MultiTimeSeriesDateSplit:
    """
    Return tuples of train_index and test_index.
    Requirements
    ------------
    1. MultiIndex DataFrame with levels named 'ticker' and 'date'
    Sources
    -------
    1. https://nander.cc/writing-custom-cross-validation-methods-grid-search
    2. https://towardsdatascience.com/time-based-cross-validation-d259b13d42b8
    3. https://hub.packtpub.com/cross-validation-strategies-for-time-series-
       forecasting-tutorial/
    """

    def __init__(
        self,
        num_folds=10,
        train_period_length=None,
        forecast_horizon=56,
        look_ahead_length=0,
    ) -> Tuple[pd.DatetimeIndex]:
        self.num_folds = num_folds
        self.lookahead_len = look_ahead_length
        self.forecast_horizon = forecast_horizon
        self.train_length = train_period_length

    def get_n_splits(
        self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray = None
    ):
        """
        Returns the number of folds in the cross-validator
        Returns
        -------
        num_folds : int
            Returns the number of folds in the cross-validator.
        """
        return self.num_folds

    def split(
        self, X: pd.DataFrame, y: np.ndarray = None, groups: np.ndarray = None
    ):
        """
        Returns folds in the cross-validator
        Returns
        -------
        fold_indexes : tuple
            Returns folds in the cross-validator.
        """
        unique_periods = X.index.get_level_values("date").unique()
        periods = sorted(unique_periods, reverse=True)

        fold_indexes = []
        for fold_num in range(self.num_folds):
            test_end_idx = fold_num * self.forecast_horizon
            test_start_idx = test_end_idx + self.forecast_horizon
            train_end_idx = test_start_idx + self.lookahead_len - 1
            if self.train_length:
                # Sliding Window
                train_start_idx = (
                    train_end_idx + self.train_length + self.lookahead_len - 1
                )
                fold_indexes.append(
                    [
                        train_start_idx,
                        train_end_idx,
                        test_start_idx,
                        test_end_idx,
                    ]
                )
            else:
                # Expanding Window
                fold_indexes.append(
                    [train_end_idx, test_start_idx, test_end_idx]
                )

        df_dates = X.reset_index()[["date"]]
        for cfg in fold_indexes:
            # Masks
            if self.train_length:
                train_start, train_end, test_start, test_end = cfg
                train_start_mask = df_dates["date"] > periods[train_start]
            else:
                train_end, test_start, test_end = cfg
            train_end_mask = df_dates["date"] <= periods[train_end]
            test_start_mask = df_dates["date"] > periods[test_start]
            test_end_mask = df_dates["date"] <= periods[test_end]
            # Slice
            if self.train_length:
                train_mask = train_start_mask & train_end_mask
                train_idx = df_dates[train_mask].index
            else:
                train_idx = df_dates[train_end_mask].index
            test_idx = df_dates[test_start_mask & test_end_mask].index
            fold_indexes = (train_idx, test_idx)
            yield fold_indexes
