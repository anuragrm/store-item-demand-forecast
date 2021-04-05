#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from time import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import src.data_custom_transformers as ct
import src.feature_utils as ut
import src.ml_helpers as mlh
import src.ml_trials_helpers as mlth
from src.cv_helpers import show_cv_dates
from src.ml_metrics import smape


def create_features(
    train_df,
    lags,
    window_size,
    used_columns,
    pred_round,
    first_date,
    gap,
    horizon,
    test_df=None,
):
    store_list = train_df["store"].unique()
    item_list = train_df["item"].unique()
    train_end_date = train_df.index.get_level_values(1).max()
    date_list = (
        pd.date_range(
            start=first_date,
            end=train_end_date + pd.DateOffset(days=gap + horizon),
        )
        .strftime("%Y-%m-%d")
        .tolist()
    )
    d = {"store": store_list, "item": item_list, "date": date_list}
    data_grid = ut.df_from_cartesian_product(d)
    data_grid["date"] = pd.to_datetime(data_grid["date"], format="%Y-%m-%d")
    start_time = time()
    data_filled = pd.merge(
        data_grid,
        train_df.reset_index().drop(columns="symbol"),
        how="left",
        on=["store", "item", "date"],
    )

    data_filled = data_filled.groupby(["store", "item"]).apply(
        lambda x: x.fillna(method="ffill").fillna(method="bfill")
    )

    add_datepart_pipe = Pipeline(
        [
            ("adddatepart", ct.DFAddDatePart("date", False, False)),
        ]
    )
    data_filled = add_datepart_pipe.fit_transform(data_filled)
    datetime_attr_cols = data_filled.columns[
        ~data_filled.columns.str.contains("|".join(used_columns))
    ].tolist()
    assert list(data_filled) == used_columns + datetime_attr_cols

    features = data_filled.groupby(["store", "item"]).apply(
        lambda x: ut.combine_features(
            x,
            ["sales"],
            lags,
            window_size,
            list(data_filled),
        )
    )
    # print(list(features))

    features.dropna(inplace=True)
    duration = time() - start_time

    if not test_df.empty:
        show_cv_dates(pred_round, train_df, test_df, duration)

    # if pred_round == 0:
    #     # display(data_filled)
    #     display(features)

    return features, train_end_date


def create_features_train_predict(
    data_train,
    lags,
    window_size,
    used_columns,
    categ_fea,
    fold_num,
    first_date,
    gap,
    horizon,
    data_test,
    model_params,
    model_fit_params,
    scoring_func,
):
    features, train_end_date = create_features(
        data_train,
        lags,
        window_size,
        used_columns,
        fold_num,
        first_date,
        gap,
        horizon,
        data_test,
    )
    (
        train_fea,
        test_fea,
        test_start_date_manual,
        _,
        test_end_date_manual,
    ) = mlh.train_test_gap_split(train_end_date, gap, features)
    if not data_test.empty:
        test_start_date_manual_check_value = data_test.index.get_level_values(
            1
        ).min()
        test_end_date_manual_check_value = data_test.index.get_level_values(
            1
        ).max()
        assert (
            test_start_date_manual
            == test_start_date_manual_check_value.strftime("%Y-%m-%d")
        )
        assert (
            test_end_date_manual
            == test_end_date_manual_check_value.strftime("%Y-%m-%d")
        )

    X_train, y_train, X_test = mlh.get_xy(train_fea, test_fea, "sales")
    y_train = mlh.transform_target(y_train)
    X_train, X_test, cats_enc, non_cats_enc = mlh.preprocess_features(
        X_train, X_test, categ_fea
    )

    feature_cols = mlh.get_feature_cols(X_train, cats_enc, non_cats_enc)

    # Use model_train_predict_sklearn() or model_train_predict()
    y_pred_test, model = mlth.model_train_predict(
        X_train,
        X_test,
        y_train,
        feature_cols,
        model_params,
        model_fit_params,
        scoring_func,
    )
    return [
        X_test,
        y_pred_test,
        model,
        train_end_date,
        test_start_date_manual,
        test_end_date_manual,
    ]


def score_model(
    data,
    cv,
    lags,
    window_size,
    used_columns,
    categ_fea,
    first_date,
    gap,
    horizon,
    model_params,
    scoring_func,
    model_fit_params={"early_stoppin_rounds": 200, "verbose": 0},
):
    num_splits = cv.get_n_splits(X=data.iloc[:2], y=data.iloc[2])
    scoring_records_summary = []
    for r, (train_idx, test_idx) in enumerate(cv.split(X=data)):
        train_df = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        print("---------- Round " + str(r + 1) + " ----------")

        (
            X_test,
            y_pred_test,
            _,
            train_cv_fold_end_date,
            test_cv_fold_start_date,
            test_cv_fold_end_date,
        ) = create_features_train_predict(
            train_df,
            lags,
            window_size,
            used_columns,
            categ_fea,
            r + 1,
            first_date,
            gap,
            horizon,
            test_df,
            model_params,
            model_fit_params,
            scoring_func,
        )
        smape_score_test = smape(
            np.expm1(y_pred_test), test_df["sales"].to_numpy()
        )
        print("SMAPE of the predictions is {}".format(smape_score_test))
        summary_dict = {
            "fold": r + 1,
            "model_params": model_params,
            "model_params_str": str(model_params),
            "train_end_date": train_cv_fold_end_date,
            "test_start_date": test_cv_fold_start_date,
            "test_end_date": test_cv_fold_end_date,
            "smape": smape_score_test,
        }
        if num_splits == 1:
            test_index = test_df.index
            y_pred_test = pd.Series(np.expm1(y_pred_test), index=test_index)
            summary_dict.update({"y_pred": y_pred_test, "X_test": X_test})
        scoring_records_summary.append(summary_dict)
    return scoring_records_summary
