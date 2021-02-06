
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
                              attrs: List[str] = ['slope', 'intercept']):
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


def fourier_coefficients(values, threshold=0.1):
    rfft = np.fft.rfft(values)
    psd = np.real(rfft * np.conj(rfft))
    # maximum, minimum = psd.max(), psd.min()
    # psd = (psd - m/inimum)/(maximum-minimum)
    # freqs = np.arange(len(psd))[psd > threshold]
    real_rfft = np.real(rfft)
    imag_rfft = np.imag(rfft)
    stats = {'rfft_real_mean': real_rfft.mean(),
             'rfft_real_std': real_rfft.std(),
             'maximum_freq': np.argmax(psd),
             'power_spectrum_mean': psd.mean(),
             'power_spectrum_q0.1': np.quantile(psd, 0.1),
             'power_spectrum_q0.5': np.quantile(psd, 0.5),
             'power_spectrum_q0.9': np.quantile(psd, 0.9),
             'power_spectrum_std': psd.std(),
             'rfft_imag_mean': imag_rfft.mean(),
             'rfft_imag_std': imag_rfft.std()}
    return stats


def calculate_features(data: pd.DataFrame,
                       timestep=np.nan) -> Dict[str, float]:
    features = defaultdict(dict)
    features['timedelta'] = timestep
    # mean and std
    data = data.fillna(method='ffill').fillna(method='backfill')
    for feature, values in data.items():
        feature_dict = defaultdict(dict)
        non_nan_values = values.dropna()
        # mean and std
        if len(non_nan_values) == 0:
            continue
        for hours in [1*60, 5*60, 10*60, 48*60]:
            last_hour_values = values.iloc[-hours:]
            mean_std = last_hour_values.agg(('mean', 'std')).to_dict()
            feature_dict[f'{hours//60}h'].update(mean_std)
        # linear features
        for hours in [48*60]:
            last_hour_values = values.iloc[-hours:]
            linear_properties = calculate_linreg_features(last_hour_values)
            feature_dict[f'{hours//60}h'].update(linear_properties)
            if feature.startswith(('bx_', 'by_', 'bz_')):
                abs_last_hour_values = last_hour_values.abs()
                abs_linear_properties = calculate_linreg_features(abs_last_hour_values)
                feature_dict[f'{hours//60}h']['abs'] = abs_linear_properties

        for hours in [48*60]:
            last_hour_values = values.iloc[-hours:]
            dx_features = calculate_dx_features(values)
            feature_dict[f'{hours//60}h'].update(dx_features)
            points_out_range = (np.abs(last_hour_values - feature_dict['48h']['mean']) >= feature_dict['48h']['std'])
            feature_dict[f'{hours//60}h']['consecutive_out_range'] = max(consecutive_count(points_out_range)) / len(points_out_range)

        # fourier
        last_hour_values = values.iloc[-72*60:]
        fourier_features = fourier_coefficients(last_hour_values)
        feature_dict['72h'].update(fourier_features)
        features[feature] = feature_dict
    return join_multiple_dict(features)
