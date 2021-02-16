from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_numeric_features(data: pd.DataFrame) -> List[str]:
    return [feature
            for feature, values in data.items()
            if is_numeric_dtype(values)]


def filter_by_pattern(features: List[str], patterns: List[str]):
    return [feature
            for feature in features
            if any(re.search(pattern, feature) is not None
                   for pattern in patterns)]


class NoOp(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X


class FeatureFilter(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self.features = filter_by_pattern(X.columns, self.patterns)
        return self

    def transform(self, X: pd.DataFrame):
        return X[:, self.features]


class Normalize(BaseEstimator, TransformerMixin):
    """A sklearn-Pipeline for normalizing DataFrames"""
    def __init__(self, drop_features: List[str] = ['timedelta', 't0',
                                                   't1', 'period']):
        self.drop_features = drop_features
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self.features = [f for f in X.columns
                         if f not in self.drop_features]

        self.scaler = self.scaler.fit(X.loc[:, self.features])
        return self

    def transform(self, X: pd.DataFrame):
        X.loc[:, self.features] = self.scaler.transform(X.loc[:, self.features])
        return X


class ToDtype(BaseEstimator, TransformerMixin):
    def __init__(self, drop_features: List[str] = ['timedelta', 'period']):
        self.drop_features = drop_features

    def fit(self, X: pd.DataFrame, y=None):
        self.features = [f for f in X.columns
                         if f not in self.drop_features]

    def transform(self, X):
        features = [f for f in X.columns if f in self.features]
        X.loc[:, features] = X.loc[:, features].astype(np.float32)
        return X


class FillNaN(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        numeric_features = get_numeric_features(X)
        self.fill_values = {feature: values.median()
                            for feature, values in X[numeric_features].items()
                            if values.isna().any()}
        return self

    def transform(self, X: pd.DataFrame):
        return X.fillna(self.fill_values)


class MakeSureFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self.features = X.columns
        return self

    def transform(self, X):
        return X.assign(**{f: np.nan
                           for f in self.features
                           if f not in X})
