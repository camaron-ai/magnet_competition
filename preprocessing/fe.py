
from itertools import groupby
import numpy as np
from scipy.stats import linregress
import pandas as pd
from typing import List, Dict
from load_data import join_multiple_dict
from collections import defaultdict


def consecutive_count(sequence: np.ndarray):
    output = [sum(seq) for value, seq in groupby(sequence) if value == 1]
    if len(output) == 0:
        output.append(0)
    return output


def consecutive_count_above_below_mean(values: np.ndarray):
    mean = np.nanmean(values)
    count_above_mean = np.nanmax(consecutive_count(values > mean))
    count_below_mean = np.nanmax(consecutive_count(values < mean))
    return {'count_above_mean': count_above_mean,
            'count_below_mean': count_below_mean}


def calculate_linreg_features(values: np.array,
                              attrs: List[str] = ['slope', 'intercept',
                                                  'stderr']):
    linreg = linregress(np.arange(len(values)), values)
    return {attr: getattr(linreg, attr)
            for attr in attrs}


def calculate_dx_features(values: np.ndarray):
    output = {}
    difference = np.square(values - np.roll(values, 1))[1:]
    output['cid_ce'] = np.sqrt(np.nanmean(difference))
    # output['d_q0.9'] = np.nanquantile(difference, 0.9)
    # output['time_since_d_q0.9'] = np.nanargmax((difference >= output['d_q0.9'])[::-1])
    return output


def time_since_peak(values):
    try:
        return {'peak': np.nanmax(np.abs(values)),
                'time_since_peak': 1/60 * np.nanargmax(values[::-1])}
    except ValueError:
        return {}


def calculate_features(data: pd.DataFrame,
                       timestep=np.nan) -> Dict[str, float]:
    features = defaultdict(dict)
    features['timedelta'] = timestep

    # mean and std
    for hours in [1*60, 5*60, 10*60, 48*60]:
        last_hours_data = data.iloc[-hours:, :]
        agg_features = last_hours_data.agg(('mean', 'std')).to_dict()
        features[f'{hours//60}h'] = agg_features

    # linear features
    for feature, values in last_hours_data.items():
        values = values.dropna()
        if len(values) == 0:
            continue
        for hours in [10*60]:
            last_hours_data = data.iloc[-hours:, :]
            linear_properties = calculate_linreg_features(values)
            features[f'{hours//60}h'][feature].update(linear_properties)

    for hours in [10*60]:
        last_hours_data = data.iloc[-hours:, :]
        for feature, values in last_hours_data.items():
            dx_features = calculate_dx_features(values)
            dx_features.update(consecutive_count_above_below_mean(values))
            if feature in ['speed', 'temperature', 'bt']:
                dx_features.update(time_since_peak(values))
            features[f'{hours//60}h'][feature].update(dx_features)
    return join_multiple_dict(features)
