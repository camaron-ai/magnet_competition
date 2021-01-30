
from itertools import groupby
import numpy as np
from scipy.stats import linregress
import pandas as pd
from typing import List, Dict
from load_data import join_multiple_dict


def consecutive_count(sequence: np.ndarray):
    output = [sum(seq) for value, seq in groupby(sequence) if value == 1]
    if len(output) == 0:
        output.append(0)
    return output


def consecutive_count_above_below_mean(values: np.ndarray):
    mean = values.mean()
    count_above_mean = max(consecutive_count(values > mean))
    count_below_mean = max(consecutive_count(values < mean))
    return {'count_above_mean': count_above_mean,
            'count_below_mean': count_below_mean}


def calculate_linreg_features(values: np.array,
                              attrs: List[str] = ['slope', 'intercept',
                                                  'stderr']):
    linreg = linregress(np.arange(len(values)), values)
    return {attr: getattr(linreg, attr)
            for attr in attrs}


def cid_ce(values: np.array):
    return np.sqrt(np.square(values - np.roll(values, 1))[1:].mean())


def calculate_features(data: pd.DataFrame,
                       timestep=np.nan) -> Dict[str, float]:
    features = {'timedelta': timestep}
    for hours in [7, 48]:
        last_hours_data = data.iloc[-hours:, :]
        agg_features = last_hours_data.agg(('mean', 'std')).to_dict()
        features[f'{hours}h'] = agg_features

    # last_2day = data.iloc[-48*60:, :]
    # complex_features = last_2day.agg(('mean', 'std')).to_dict()
    # for feature, values in last_2day.items():
    #     values.dropna(inplace=True)
    #     if len(values) == 0:
    #         values = np.array([np.nan])
    #     _complex_features = calculate_linreg_features(values)
    #     _complex_features['cid_ce'] = cid_ce(values)
    #     _complex_features.update(consecutive_count_above_below_mean(values))
    #     complex_features[feature].update(_complex_features)
    # features['48h'] = complex_features
    return join_multiple_dict(features)
