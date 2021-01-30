from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from joblib import Parallel, delayed
from preprocessing.fe import calculate_features


def fillna_features(X: pd.DataFrame,
                    features: List[str] = None,
                    interpolate: bool = True):
    if features is None:
        features = X.columns
    # fillnan values with the closest non-nan value
    for feature, values in X[features].items():
        if is_numeric_dtype(values) and values.isna().any():
            values = (values.interpolate()
                      if interpolate else
                      values.fillna(method='ffill'))
            values = values.fillna(method='backfill')
            X[feature] = values
    return X


def create_target(dst_values: pd.DataFrame):
    target = dst_values.loc[:, ['period', 'timedelta']].reset_index(drop=True)
    target['t0'] = dst_values['dst'].values
    target['t1'] = target.groupby('period')['t0'].shift(-1).fillna(-12)
    return target


def merge_daily(data: pd.DataFrame,
                other: pd.DataFrame) -> pd.DataFrame:
    data.loc[:, 'day'] = data['timedelta'].dt.days
    other.loc[:, 'day'] = other['timedelta'].dt.days
    other.drop_duplicates(subset=['period', 'day'],
                          inplace=True)
    data = data.merge(other.drop('timedelta', axis=1),
                      on=['period', 'day'],
                      how='left')
    data = fillna_features(data,
                           features=other.columns,
                           interpolate=False)
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
        data.loc[:, f'{feature}_direction'] = np.clip(direction, -1, 1)
    return data


def solar_wind_preprocessing(solar_wind: pd.DataFrame):
    solar_wind.loc[:, 'temperature'] = np.log(solar_wind['temperature'] + 1)
    solar_wind.loc[:, 'speed'] = np.sqrt(solar_wind['speed'])

    return solar_wind


def split_data_in_chunks(data: pd.DataFrame,
                         stride=pd.to_timedelta(7, unit='d')
                         ) -> Dict[Tuple[str], pd.DataFrame]:
    one_minute = pd.to_timedelta(1, unit='m')
    output = {}
    for timestep in data.index.ceil('H').unique():
        output[timestep] = data.loc[timestep-stride: timestep, :]
    return output


# def from_chunks_to_dataframe(chunks: Dict[str, pd.DataFrame], n_jobs: int = 8):
#     return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(calculate_features)(datastep, timestep)
#                         for timestep, datastep in chunks.items()))

# def from_chunks_to_dataframe(chunks: Dict[str, pd.DataFrame], n_jobs: int = 8):
#     return pd.DataFrame([calculate_features(datastep, timestep)
#                         for timestep, datastep in chunks.items()])


def one_chunk_to_dataframe(chunk: pd.DataFrame,
                           timestep=np.nan):
    features = calculate_features(chunk, timestep)
    return pd.DataFrame([features])


def split_into_period(data: pd.DataFrame, features: List[str],
                      n_jobs: int = 8) -> pd.DataFrame:
    output_data = []
    for period, period_data in data.groupby('period'):
        chunks = split_data_in_chunks(period_data.loc[:, features])
        fe_data = from_chunks_to_dataframe(chunks, n_jobs=n_jobs)
        fe_data.loc[:, 'period'] = period
        output_data.append(fe_data)
        del chunks
    return pd.concat(output_data, ignore_index=True, axis=0)
