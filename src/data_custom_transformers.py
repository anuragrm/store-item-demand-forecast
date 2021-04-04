#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sklearn.base import TransformerMixin

import src.eng_features_helpers as fh


class DFAddDatePart(TransformerMixin):
    def __init__(
        self,
        date_col_name="date",
        drop=False,
        inplace=False,
    ):
        self.date_col_name = date_col_name
        self.drop = drop
        self.inplace = inplace

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X = fh.add_datepart(
            X,
            self.date_col_name,
            drop=self.drop,
            inplace=self.inplace,
        )
        return X


class DFMultiColSort(TransformerMixin):
    def __init__(self, cols_to_sort, cols_sort_asc=[True, True, True]):
        self.cols_to_sort = cols_to_sort
        self.cols_sort_asc = cols_sort_asc

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X.sort_values(
            self.cols_to_sort, ascending=self.cols_sort_asc, inplace=True
        )
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self = self.fit(X, y)
        return self.transform(X)
