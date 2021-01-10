from typing import Tuple, List
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def fillna_features(X: pd.DataFrame):
    # fillnan values with the closest non-nan value
    for feature, values in X.items():
        if is_numeric_dtype(values) and values.isna().any():
            X[feature] = values.fillna(method='ffill').fillna(method='backfill')
    return X


def agregate_data(X: pd.DataFrame, on: List[str],
                  features: List[str],
                  agg_attr: Tuple[str] = ('mean', 'std', 'min', 'max')):
    aggr_data = X.groupby(on)[features].agg(agg_attr)
    aggr_data.columns = ['_'.join(column) for column in aggr_data.columns]
    aggr_data.reset_index(inplace=True)
    return aggr_data


def create_target(dst_values: pd.DataFrame):
    target = dst_values.loc[:, ['period', 'timedelta']].reset_index(drop=True)
    target['t0'] = dst_values['dst'].values
    target['t1'] = target.groupby('period')['t0'].shift(-1).fillna(-12)
    return target


def merge_sunspots(data: pd.DataFrame,
                   sunspots: pd.DataFrame) -> pd.DataFrame:
    data['day'] = data['timedelta'].dt.days
    sunspots['day'] = sunspots['timedelta'].dt.days
    data = data.merge(sunspots.drop('timedelta', axis=1),
                      on=['period', 'day'],
                      how='left')
    data.drop('day', inplace=True, axis=1)
    sunspots.drop('day', inplace=True, axis=1)
    return data


def preprocessing(solar_wind: pd.DataFrame,
                  sunspots: pd.DataFrame,
                  features: List[str] = None):
    if features is None:
        features = solar_wind.columns.drop(['timedelta', 'period', 'source'])

    # adding a minute and ceiling the minutes to the next hour to not leak
    # about the future
    solar_wind['timedelta'] += pd.to_timedelta(1, unit='m')
    solar_wind['timedelta'] = solar_wind['timedelta'].dt.ceil('H')
    hourly_solar_wind = agregate_data(solar_wind,
                                      on=['period', 'timedelta'],
                                      features=features)

    data = merge_sunspots(hourly_solar_wind, sunspots)
    data = fillna_features(data)
    return data
