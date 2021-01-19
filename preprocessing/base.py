from typing import Tuple, List
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def fillna_features(X: pd.DataFrame):
    # fillnan values with the closest non-nan value
    for feature, values in X.items():
        if is_numeric_dtype(values) and values.isna().any():
            values = values.fillna(method='ffill').fillna(method='backfill')
            X[feature] = values
    return X


def agregate_data(X: pd.DataFrame, on: List[str],
                  features: List[str],
                  agg_attr: Tuple[str] = ('mean', 'std')):
    aggr_data = X.groupby(on)[features].agg(agg_attr)
    aggr_data.columns = ['_'.join(column) for column in aggr_data.columns]
    aggr_data.reset_index(inplace=True)
    return aggr_data


def create_target(dst_values: pd.DataFrame):
    target = dst_values.loc[:, ['period', 'timedelta']].reset_index(drop=True)
    target['t0'] = dst_values['dst'].values
    target['t1'] = target.groupby('period')['t0'].shift(-1).fillna(-12)
    return target


def merge_daily(data: pd.DataFrame,
                other: pd.DataFrame) -> pd.DataFrame:
    data['day'] = data['timedelta'].dt.days
    other['day'] = other['timedelta'].dt.days
    other.drop_duplicates(subset=['period', 'day'],
                          inplace=True)
    data = data.merge(other.drop('timedelta', axis=1),
                      on=['period', 'day'],
                      how='left')
    data.drop('day', inplace=True, axis=1)
    other.drop('day', inplace=True, axis=1)
    return data


def stl_preprocessing(data: pd.DataFrame):
    to_drop = ['gse_x_dscovr', 'gse_y_dscovr', 'gse_z_dscovr']
    data.drop(to_drop, inplace=True, axis=1)
    working_features = ['gse_x_ace', 'gse_y_ace', 'gse_z_ace']
    period_data = data.groupby('period')
    for feature in working_features:
        direction = period_data[feature].diff().fillna(0)
        data[f'{feature}_direction'] = np.clip(direction, -1, 1)
    return data


def calculate_magnitud(vectors):
    return np.sqrt(np.square(vectors).sum(axis=1))


def solar_wind_preprocessing(solar_wind: pd.DataFrame,
                             features: List[str]):
    solar_wind['timedelta'] += pd.to_timedelta(1, unit='m')
    solar_wind['timedelta'] = solar_wind['timedelta'].dt.ceil('H')
    solar_wind['temperature'] = np.log(solar_wind['temperature'] + 1)
    # solar_wind['byz'] = calculate_magnitud(solar_wind[['by_gsm', 'bz_gsm']])
    # features.append('byz')
    return solar_wind, features


def preprocessing(solar_wind: pd.DataFrame,
                  sunspots: pd.DataFrame,
                  stl_pos: pd.DataFrame = None,
                  features: List[str] = None):
    if features is None:
        features = solar_wind.columns.drop(['timedelta', 'period', 'source'])

    # adding a minute and ceiling the minutes to the next hour to not leak
    # about the future
    solar_wind, features = solar_wind_preprocessing(solar_wind, features)
    hourly_solar_wind = agregate_data(solar_wind,
                                      on=['period', 'timedelta'],
                                      features=features)

    data = merge_daily(hourly_solar_wind, sunspots)
    if stl_pos is not None:
        stl_pos = stl_preprocessing(stl_pos)
        data = merge_daily(data, stl_pos)
    data = fillna_features(data)
    return data
