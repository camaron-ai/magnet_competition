
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
    difference = (values - np.roll(values, 1))[1:]
    non_zero = np.nonzero(values[:-1])
    output['cid_ce'] = np.sqrt(np.nanmean(np.square(difference)))
    output['change_rate'] = np.nanmean(difference[non_zero] / values[:-1][non_zero])
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
    psd = np.real(rfft * np.conj(rfft)) / (len(values)**2)
    real_rfft = np.real(rfft) / len(values)
    imag_rfft = np.imag(rfft) / len(values)
    stats = {'rfft_real_mean': real_rfft.mean(),
             'rfft_real_std': real_rfft.std(),
             'power_spectrum_mean': psd.mean(),
             'power_spectrum_q0.1': np.quantile(psd, 0.1),
             'power_spectrum_q0.5': np.quantile(psd, 0.5),
             'power_spectrum_q0.9': np.quantile(psd, 0.9),
             'power_spectrum_std': psd.std(),
             'rfft_imag_mean': imag_rfft.mean(),
             'rfft_imag_std': imag_rfft.std()}
    return stats


def time_iter(data, periods):
    for period in periods:
        yield period, data.iloc[-period:]


def point_in_range(values, mean, std, p=1.96):
    points_in_range = (np.abs(values - mean) <= p*std)
    max_consecutive = max(consecutive_count(points_in_range))
    max_consecutive /= len(points_in_range)
    return {'consecutive_in_range': max_consecutive}


mean_std_periods = [1*60, 5*60, 10*60, 48*60]
lin_periods = [48*60]
outlier_periods = [48*60]
fourier_periods = [72*60]


def _calculate_features(values: pd.Series,
                        compute_abs_lin=False):
    feature_dict = defaultdict(dict)
    imputed_values = values.fillna(method='ffill').fillna(method='backfill')
    non_nan_values = values.dropna()
    # if there is not data, return empty dict
    if len(non_nan_values) == 0:
        return feature_dict
    for hours, last_n_values in time_iter(values, mean_std_periods):
        mean_std = last_n_values.agg(('mean', 'std')).to_dict()
        feature_dict[f'{hours//60}h'].update(mean_std)

    # linear features
    for hours, last_n_values in time_iter(values, lin_periods):
        last_n_values = last_n_values.dropna()
        if len(last_n_values) < 1:
            continue
        linear_properties = calculate_linreg_features(last_n_values)
        feature_dict[f'{hours//60}h'].update(linear_properties)
        if compute_abs_lin:
            abs_last_n_values = last_n_values.abs()
            abs_linear_properties = calculate_linreg_features(abs_last_n_values)
            feature_dict[f'{hours//60}h']['abs'] = abs_linear_properties

    for hours, last_n_values in time_iter(values, outlier_periods):
        last_n_values = last_n_values.to_numpy()
        dx_features = calculate_dx_features(last_n_values)
        pin_range = point_in_range(last_n_values,
                                     mean=feature_dict['48h']['mean'],
                                     std=feature_dict['48h']['std'])
        feature_dict[f'{hours//60}h'].update(dx_features)
        feature_dict[f'{hours//60}h'].update(pin_range)
    # fourier
    for hours, last_n_values in time_iter(imputed_values, fourier_periods):
        last_n_values = last_n_values.to_numpy()
        fourier_features = fourier_coefficients(last_n_values)
        feature_dict[f'{hours//60}h'].update(fourier_features)
    return feature_dict




def calculate_features(data: pd.DataFrame,
                       timestep=np.nan) -> Dict[str, float]:
    features = defaultdict(dict)
    features['timedelta'] = timestep
    for feature, values in data.items():
        is_coor = feature.startswith(('bx_', 'by_', 'bz_'))
        features[feature] = _calculate_features(values, is_coor)
    return join_multiple_dict(features)
