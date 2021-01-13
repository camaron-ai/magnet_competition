from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from itertools import product
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import re


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


class Lagger(FeatureFilter):
    def __init__(self, lags: List[int], patterns: List[str] = None,
                 time_feature: str = 'timedelta',
                 on: List[str] = ['period'], dropna: bool = False,
                 unit: str = 'H'):
        self.on = on
        self.patterns = patterns
        self.time_feature = time_feature
        self.lags = lags
        self.dropna = dropna
        self.unit = unit
        self.on.append(self.time_feature)

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

        X.fillna({feature: -999
                  for feature, values in X.items()
                  if (is_numeric_dtype(values) and
                      values.isna().any())}, inplace=True)
        return X


class RollingStats(FeatureFilter):
    def __init__(self, windows: List[int], patterns: List[str] = None,
                 attr: List[str] = ['mean'],
                 on: List[str] = ['period'], dropna: bool = False):
        self.on = on
        self.attr = attr
        self.patterns = patterns
        self.windows = windows
        self.dropna = dropna

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



class DifferenceFeatures(FeatureFilter):
    def __init__(self, patterns: List[str] = None,
                 on: List[str] = ['period']):
        self.patterns = patterns
        self.on = on

    def transform(self, X: pd.DataFrame):
        difference = X.groupby(self.on)[self.features].diff()
        difference.fillna(0, inplace=True)
        difference.rename(columns={feature: f'{feature}_diff'
                                   for feature in self.features},
                          inplace=True)
        X = pd.concat([X, difference], axis=1)
        return X


class DropFeatureByCorr(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.9,
                 patterns: List[str] = ['t0', 't1']):
        self.patterns = patterns
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        target = filter_by_pattern(X.columns, self.patterns)
        features = X.columns.drop(target).to_list()
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
                 patterns: List[str] = ['t0', 't1']):
        self.patterns = patterns
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        target = filter_by_pattern(X.columns, self.patterns)
        corr_matrix = X.corr().abs()
        features = corr_matrix.columns.drop(target)
        corr_matrix = corr_matrix.loc[features, target]
        to_drop_index = (corr_matrix < self.threshold)
        if len(to_drop_index.shape) > 1:
            to_drop_index = to_drop_index.any(axis=1)
        self.to_drop = features[to_drop_index]
        return self


class DropFeatures(FeatureFilter):
    def __init__(self, patterns: List[str]):
        self.patterns = patterns

    def transform(self, X: pd.DataFrame):
        return X.drop(self.features, axis=1)
