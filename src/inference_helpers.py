#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import src.feature_utils as ftu
from src.utils import create_features_train_predict


def show_future_prediction_data_dates(data, gap):
    train_dates = data.index.get_level_values(1)
    gap_start = train_dates.max() + pd.DateOffset(days=1)
    gap_end = gap_start + pd.DateOffset(days=gap - 1)
    train_split_size = (
        data.groupby(level="symbol").size().value_counts().index[0]
    )
    gap_size = len(pd.date_range(gap_start.date(), gap_end.date()))
    print(
        f"train = {train_dates.min().date()} - {train_dates.max().date()} "
        f"({train_split_size}) | "
        f"gap = {gap_start.date()} - {gap_end.date()} ({gap_size})"
    )


def get_future_prediction_data_grid(data, gap, horizon):
    store_list = data["store"].unique()
    item_list = data["item"].unique()
    train_end_date = data.index.get_level_values(1).max()
    date_list = pd.date_range(
        start=train_end_date + pd.DateOffset(days=gap + 1),
        end=train_end_date + pd.DateOffset(days=gap + horizon),
    )
    d = {"store": store_list, "item": item_list, "date": date_list}
    data_grid = ftu.df_from_cartesian_product(d)
    data_grid = data_grid.assign(
        symbol=data_grid["store"].astype(str)
        + "_"
        + data_grid["item"].astype(str)
    ).set_index(["symbol", "date"])
    return data_grid


def predict_future(
    r,
    data_train,
    lags,
    window_size,
    used_columns,
    categ_fea,
    first_date,
    gap,
    horizon,
    data_test,
    model_params,
    scoring_func,
    model_fit_params={"early_stoppin_rounds": 200, "verbose": 0},
):
    show_future_prediction_data_dates(data_train, gap)

    _, y_pred_test, trained_model = create_features_train_predict(
        data_train,
        lags,
        window_size,
        used_columns,
        categ_fea,
        r,
        first_date,
        gap,
        horizon,
        data_test,
        model_params,
        model_fit_params,
        scoring_func,
    )
    df_future_pred = get_future_prediction_data_grid(data_train, gap, horizon)
    df_future_pred["pred"] = np.expm1(y_pred_test)
    return [df_future_pred, trained_model]
