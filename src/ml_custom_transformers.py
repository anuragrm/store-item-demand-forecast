#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DFOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    dff = DataFrame(
        {
            "pets": ["cat", "dog", "cat", "monkey", "dog", "dog"],
            "owner": ["Champ", "Ron", "Brick", "Champ", "Veronica", "Ron"],
            "location": ["SD", "NY", "NY", "SD", "SD", "NY"],
        }
    )
    train = dff.iloc[:4, :]
    test = dff.iloc[4:, :]
    import src.custom_transformers as sct
    ohe = sct.DFOneHotEncoder()
    ohe = ohe.fit(train)
    test_tr = DataFrame(ohe.transform(test), columns=ohe.get_feature_names())
    test_recovered = DataFrame(
        ohe.inverse_transform(test_tr.astype(int)), columns=list(test)
    )
    """

    def __init__(self):
        self.ohe = None

    def fit(self, X, y=None):
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.ohe.fit(X)
        cdt = dict(zip([f"x{c}" for c in range(X.shape[1])], list(X)))
        self._feature_names = [
            c.replace(c.split("_")[0], cdt[c.split("_")[0]]).replace("_", "=")
            for c in list(self.ohe.get_feature_names())
        ]
        return self

    def transform(self, X):
        Xohe = pd.DataFrame(
            self.ohe.transform(X).toarray(), columns=self._feature_names
        )
        return Xohe

    def get_feature_names(self):
        return self._feature_names

    def inverse_transform(self, X):
        Xiohe = self.ohe.inverse_transform(X)
        return Xiohe


class DFLog1p(TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X[self.col_name] = np.log1p(X[self.col_name])
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)


class DFStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sc = None

    def fit(self, X, y=None):
        self.sc = StandardScaler()
        self.sc.fit(X)
        return self

    def transform(self, X):
        X_sc = pd.DataFrame(
            self.sc.transform(X), index=X.index, columns=X.columns
        )
        return X_sc
