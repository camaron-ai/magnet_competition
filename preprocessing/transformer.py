from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from itertools import product
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np


def get_numeric_features(data: pd.DataFrame) -> List[str]:
    return [feature
            for feature, values in data.items()
            if is_numeric_dtype(values)]


class NoOp(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X


class Lagger(BaseEstimator, TransformerMixin):
    def __init__(self, lags: List[int], features: List[str] = None,
                 time_feature: str = 'timedelta',
                 on: List[str] = ['period'], dropna: bool = False,
                 unit: str = 'H'):
        self.on = on
        self.features = features
        self.time_feature = time_feature
        self.lags = lags
        self.dropna = dropna
        self.unit = unit
        self.on.append(self.time_feature)

    def fit(self, X: pd.DataFrame, y=None):
        if self.features is None:
            self.features = get_numeric_features(X)
        return self

    def transform(self, X: pd.DataFrame):
        X_delta = X.loc[:, self.on + self.features].copy(deep=True)
        for lag in self.lags:
            time_feature = X.loc[:, self.time_feature].copy(deep=True)
            time_feature += pd.to_timedelta(lag, unit=self.unit)
            X_delta.loc[:, self.time_feature] = time_feature
            X = X.merge(X_delta, on=self.on, how='left',
                        suffixes=('', f'_{lag}_lag'))
        if self.dropna:
            X.dropna(inplace=True)
            X.reset_index(drop=True, inplace=True)
        return X


class RollingStats(BaseEstimator, TransformerMixin):
    def __init__(self, windows: List[int], features: List[str] = None,
                 attr: List[str] = ['mean'],
                 on: List[str] = ['period'], dropna: bool = False):
        self.on = on
        self.attr = attr
        self.features = features
        self.windows = windows
        self.dropna = dropna

    def fit(self, X: pd.DataFrame, y=None):
        if self.features is None:
            self.features = get_numeric_features(X)
        return self

    def rolling_series(self, window: int,
                       attr: str = 'mean'):
        def _inner(series):
            moving_series = series.rolling(window, min_periods=1)
            attr_mv_series = getattr(moving_series, attr)
            return attr_mv_series()
        return _inner

    def transform(self, X: pd.DataFrame):
        for window, feature, attr in product(self.windows,
                                             self.features, self.attr):
            grouped_feature = X.groupby(self.on)[feature]
            moving_series = grouped_feature.transform(self.rolling_series(window, attr))
            moving_series.fillna(0, inplace=True)
            X[f'{feature}_{attr}_{window}h'] = moving_series
        if self.dropna:
            X.dropna(inplace=True)
            X.reset_index(drop=True, inplace=True)
        return X


class DropFeatureByCorr(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.9,
                 target: List[str] = ['t0', 't1']):
        self.target = target
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        features = X.columns.drop(self.target).to_list()
        corr_matrix = X.loc[:, features].corr().abs()
        upper_triu_index = np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)
        upper_triu = corr_matrix.where(upper_triu_index)
        self.to_drop = [feature
                        for feature in upper_triu.columns
                        if (upper_triu[feature] > self.threshold).any()]
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(self.to_drop, axis=1)


class DropFeaturesByCorrTarget(DropFeatureByCorr):
    def __init__(self, threshold: float = 0.05,
                 target: List[str] = ['t0', 't1']):
        self.target = target
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        corr_matrix = X.corr().abs()
        features = corr_matrix.columns.drop(self.target)
        corr_matrix = corr_matrix.loc[features, self.target]
        to_drop_index = (corr_matrix < self.threshold)
        if len(to_drop_index.shape) > 1:
            to_drop_index = to_drop_index.any(axis=1)
        self.to_drop = features[to_drop_index]
        return self

