#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from time import time

import lightgbm as lgb


def train_lgbm_sklearn(
    model_params,
    model_fit_params,
    X_train,
    y_train,
    watchlist_sklearn,
    model_scoring_func,
    feature_cols,
):
    print("Training LightGBM model started.")
    tr_start_time = time()
    lgbmreg_pipe = lgb.LGBMRegressor(**model_params)
    lgbmreg_pipe.fit(
        X_train,
        y_train,
        early_stopping_rounds=model_fit_params["early_stopping_rounds"],
        eval_names=["train", "eval"],
        eval_set=watchlist_sklearn,
        eval_metric=model_scoring_func,
        feature_name=feature_cols,
        verbose=model_fit_params["verbose"],
    )
    tr_duration = time() - tr_start_time
    print(f"Training LightGBM model finished in {tr_duration:.2f} seconds.")
    return lgbmreg_pipe


def model_train_predict_sklearn(
    X_train,
    X_test,
    y_train,
    feature_cols,
    model_params,
    model_fit_params,
    scoring_func,
):
    watchlist_sklearn = [(X_train, y_train)]
    model = train_lgbm_sklearn(
        model_params,
        model_fit_params,
        X_train,
        y_train,
        watchlist_sklearn,
        scoring_func,
        feature_cols,
    )
    y_pred_test = model.predict(X_test)
    print("Prediction made")
    return [y_pred_test, model]


def train_lgbm(
    model_params, model_fit_params, lgbtrain, watchlist, model_scoring_func
):
    print("Training LightGBM model started.")
    tr_start_time = time()
    model = lgb.train(
        model_params,
        lgbtrain,
        num_boost_round=model_fit_params["num_boost_round"],
        early_stopping_rounds=model_fit_params["early_stopping_rounds"],
        feval=model_scoring_func,
        # categorical_feature=categ_fea,
        valid_sets=watchlist,
        valid_names=["train", "eval"],
        verbose_eval=200,
    )
    tr_duration = time() - tr_start_time
    print(f"Training LightGBM model finished in {tr_duration:.2f} seconds.")
    return model


def model_train_predict(
    X_train,
    X_test,
    y_train,
    feature_cols,
    model_params,
    model_fit_params,
    scoring_func,
):
    lgbtrain = lgb.Dataset(
        data=X_train,
        label=y_train,
        feature_name=feature_cols,
        # categorical_feature=categ_fea,
    )
    watchlist = [lgbtrain]

    model = train_lgbm(
        model_params, model_fit_params, lgbtrain, watchlist, scoring_func
    )
    y_pred_test = model.predict(X_test)
    print("Prediction made")
    return [y_pred_test, model]
