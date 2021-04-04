#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import src.ml_custom_transformers as mlct


def check_normality(y):
    alpha = 0.001
    _, p = stats.normaltest(y.to_numpy())
    return p < alpha


def train_test_gap_split(train_end_date, gap, features):
    train_mask = features["date"] <= train_end_date
    train_fea = features[train_mask].reset_index(drop=True)
    train_end_date_manual = max(train_fea["date"]).strftime("%Y-%m-%d")
    assert train_end_date_manual == train_end_date.strftime("%Y-%m-%d")
    test_dates_mask = features["date"] >= train_end_date + pd.DateOffset(
        days=gap + 1
    )
    test_fea = features[test_dates_mask].reset_index(drop=True)
    test_start_date_manual = min(test_fea["date"]).strftime("%Y-%m-%d")
    test_end_date_manual = max(test_fea["date"]).strftime("%Y-%m-%d")
    print(
        "Max. training date = {}, Test/Prediction Dates = {} - {}".format(
            train_end_date_manual,
            test_start_date_manual,
            test_end_date_manual,
        )
    )
    return [
        train_fea,
        test_fea,
        test_start_date_manual,
        train_end_date_manual,
        test_end_date_manual,
    ]


def get_xy(train_fea, test_fea, y_name):
    X_train = train_fea.drop([y_name, "date"], axis=1)
    y_train = train_fea[y_name]
    X_val = test_fea.drop([y_name, "date"], axis=1)
    # y_val = test_fea[y_name]
    return [X_train, y_train, X_val]


def preprocess_features(X_train, X_test, categ_fea):
    # One-Hot Encode categoricals
    categorical_transformer = mlct.DFOneHotEncoder()
    preprocessor = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categ_fea)],
        remainder="passthrough",
    )
    pipe_preprocess = Pipeline(steps=[("pp", preprocessor)])
    pipe_preprocess = pipe_preprocess.fit(X_train)
    ohe_cols_names = [
        x.replace("cat__", "")
        for x in pipe_preprocess.named_steps["pp"].get_feature_names()
    ]
    # print(ohe_cols_names)
    X_train = pd.DataFrame(
        pipe_preprocess.transform(X_train),
        columns=ohe_cols_names,
    )
    X_test = pd.DataFrame(
        pipe_preprocess.transform(X_test),
        columns=ohe_cols_names,
    )
    # print(list(X_train))
    # display(X_train.head())

    # Get lists of features
    ohe_cols_mask = X_train.columns.str.contains("=")
    cats_enc = X_train.columns[ohe_cols_mask].tolist()
    non_cats_enc = X_train.columns[~ohe_cols_mask].tolist()

    # Scaling numericals
    numerical_transformer = mlct.DFStandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[("nums", numerical_transformer, non_cats_enc)],
        remainder="passthrough",
    )
    pipe_preprocess = Pipeline(steps=[("pp", preprocessor)])
    pipe_preprocess = pipe_preprocess.fit(X_train)
    X_train = pd.DataFrame(
        pipe_preprocess.transform(X_train),
        columns=non_cats_enc + cats_enc,
    )
    X_test = pd.DataFrame(
        pipe_preprocess.transform(X_test),
        columns=non_cats_enc + cats_enc,
    )
    # print(list(X_train))
    # display(X_train.head())

    # Change datatypes after processing
    X_train[cats_enc] = X_train[cats_enc].astype(int)
    X_train[non_cats_enc] = X_train[non_cats_enc].astype(float)
    X_test[cats_enc] = X_test[cats_enc].astype(int)
    X_test[non_cats_enc] = X_test[non_cats_enc].astype(float)
    # display(X_train.dtypes.to_frame())
    return [X_train, X_test, cats_enc, non_cats_enc]


def transform_target(y):
    log1p_pipe = Pipeline([("log1p", mlct.DFLog1p("sales"))])
    y = log1p_pipe.fit_transform(y.to_frame()).squeeze()
    return y


def get_feature_cols(X_train, cats_encoded_cols, non_cats_encoded_cols):
    feature_cols = [c for c in list(X_train) if c not in ["date"]]
    assert len(list(X_train)) == len(feature_cols)
    assert list(X_train) == feature_cols
    assert len(cats_encoded_cols) + len(non_cats_encoded_cols) == len(
        feature_cols
    )
    assert set(non_cats_encoded_cols + cats_encoded_cols) == set(feature_cols)
    return feature_cols


def get_best_model_hyper_params(df_splits, metric_name):
    df_params = (
        df_splits.groupby(["model_params_str"])[metric_name]
        .mean()
        .reset_index()
        .sort_values(by=[metric_name], ascending=True)
    )
    df_splits_best = df_splits[
        df_splits["model_params_str"] == df_params.iloc[0]["model_params_str"]
    ]
    best_model_params = df_splits_best.iloc[0]["model_params"]
    return best_model_params
