from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from itertools import product
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA
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


class Lagger(FeatureFilter):
    def __init__(self, lags: List[int], patterns: List[str] = None,
                 time_feature: str = 'timedelta',
                 on: List[str] = ['period'], dropna: bool = False,
                 unit: str = 'H'):
        self.patterns = patterns
        self.time_feature = time_feature
        self.lags = lags
        self.dropna = dropna
        self.unit = unit
        self.on = on + [self.time_feature]

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

        X.fillna({feature: values.median()
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
            attr_mv_series = self.getattr(moving_series, attr)
            return attr_mv_series
        return _inner

    def getattr(self, series, attr):
        splitted_attr = attr.split('_')
        if len(splitted_attr) > 1:
            attr, param = splitted_attr
            series = getattr(series, attr)
            return series(float(param))
        else:
            return getattr(series, attr)()

    def transform(self, X: pd.DataFrame):
        for window, feature, attr in product(self.windows,
                                             self.features, self.attr):
            grouped_feature = X.groupby(self.on)[feature]
            moving_series = grouped_feature.transform(self.rolling_series(window, attr))
            moving_series.fillna(0, inplace=True)
            X.loc[:, f'{feature}_{attr}_{window}h'] = moving_series
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


def lag_values_n_times(values, lags):
    lag_values = [np.roll(values, lag)
                  for lag in reversed(range(1, lags+1))]
    return np.stack(lag_values + [values], axis=1)


class Fourier(FeatureFilter):
    def __init__(self, lags: int, top: int = None,
                 patterns: List[str] = None):
        self.lags = lags
        self.top = top
        self.patterns = patterns
        self.top_freq = {}

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X=X, y=y)
        if self.top is None:
            return self

        for feature in self.features:
            series = X.loc[:, feature].to_numpy()
            rfft = self.compute_rfft(series)
            psd = np.real(rfft * np.conj(rfft))
            freq_importances = psd.mean(axis=0)
            top_freq = np.argsort(freq_importances)[::-1][:self.top]
            self.top_freq[feature] = top_freq
        return self

    def _gen_columns(self, suffix: str, n: int):
        return [f'{suffix}_fft_coef_{c}' for c in range(n)]

    def get_columns(self, suffix, n):
        return (self._gen_columns(f'{suffix}_real', n) +
                self._gen_columns(f'{suffix}_imag', n))

    def transform(self, X: pd.DataFrame):
        output = [X]
        for feature in self.features:
            series = X.loc[:, feature].to_numpy()
            rfft = self.compute_rfft(series)
            abs_fft = np.abs(rfft)
            if self.top is not None:
                rfft = rfft[:, self.top_freq[feature]]
            real_fft = np.real(rfft)
            imag_fft = np.imag(rfft)
            columns = self.get_columns(feature, rfft.shape[1])
            fft_data = np.concatenate((real_fft, imag_fft), axis=1)
            fft_df = pd.DataFrame(fft_data,
                                  columns=columns,
                                  index=X.index)
            fft_df[f'{feature}_mean_rfft'] = abs_fft.mean(axis=1)
            fft_df[f'{feature}_std_rfft'] = abs_fft.std(axis=1)
            output.append(fft_df)
            del fft_data
        return pd.concat(output, axis=1)

    def compute_rfft(self, series):
        lag_series = lag_values_n_times(series, self.lags)
        rfft = np.fft.rfft(lag_series, axis=1)
        del lag_series
        return rfft


class CustomPCA(FeatureFilter):
    def __init__(self, n_components=None, copy=True,
                 whiten=False, svd_solver='auto', tol=0.0,
                 iterated_power='auto', random_state=None,
                 patterns: List[str] = None):
        self.n_components = n_components
        self.patterns = patterns
        self.scaler = StandardScaler()
        pca = PCA(n_components=n_components, copy=copy,
                  whiten=whiten, svd_solver=svd_solver, tol=tol,
                  iterated_power=iterated_power,
                  random_state=random_state)
        self.pca = Pipeline(steps=[('scaled', StandardScaler()),
                                   ('pca', pca)])

    def fit(self, X: pd.DataFrame, y=None):
        super().fit(X=X, y=y)
        self.pca = self.pca.fit(X.loc[:, self.features])
        return self

    def get_features_names(self, total_components):
        return [f'PCA_C{C+1}/{total_components}'
                for C in range(total_components)]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed_data = self.pca.transform(X.loc[:, self.features])
        columns = self.get_features_names(transformed_data.shape[1])
        pca_data = pd.DataFrame(transformed_data, columns=columns)
        del transformed_data
        return pd.concat([X, pca_data], axis=1)


class Normalize(BaseEstimator, TransformerMixin):
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
