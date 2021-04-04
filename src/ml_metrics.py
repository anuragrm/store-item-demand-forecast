#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    # need np.expm1(...) since target was log-scaled in step 11.
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return "SMAPE", smape_val, False


def lgbm_smape_sklearn(y, yhat):
    # need np.expm1(...) since target was log-scaled in step 11.
    smape_val = smape(np.expm1(yhat), np.expm1(y))
    return "SMAPE", smape_val, False
