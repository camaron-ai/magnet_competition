import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Tuple
from preprocessing.base import solar_wind_preprocessing
from preprocessing.base import stl_preprocessing
from preprocessing.base import merge_daily, one_chunk_to_dataframe
import logging
import joblib
import default
from models import library as model_library
from collections import defaultdict
import warnings
import torch

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


def load_models():
    """A function to load everything necessary to make a prediction"""
    # create a repository to save every artifacts
    repo = defaultdict(dict)
    # define the path where all experiments are
    path = Path('./experiments/')
    for experiment in os.listdir(path):
        # for each experiment
        experiment_path = path.joinpath(experiment, 'models')
        # if the models is not trained, skip it
        if not os.path.exists(experiment_path):
            continue
        print(f'loading experiment {experiment}')
        # load everything need it
        model_h0 = joblib.load(experiment_path / 'model_h0.pkl')

        model_h1 = (joblib.load(experiment_path / 'model_h1.pkl')
                    if os.path.exists(experiment_path / 'model_h1.pkl')
                    else None)
        pipeline = joblib.load(experiment_path / 'pipeline.pkl')
        # save it into the experiment's dict
        repo[experiment]['model_h0'] = model_h0
        repo[experiment]['model_h1'] = model_h1
        repo[experiment]['pipeline'] = pipeline
    # print('complete!')
    return repo


repo = load_models()


def predict(test_data, model_h0, model_h1=None):
    if model_h1 is None:
        tensor_features = torch.from_numpy(test_data.to_numpy())
        prediction_output = model_h0(tensor_features)
        prediction = prediction_output['prediction'][-1]
        pred_at_t0, pred_at_t1 = prediction
        pred_at_t0, pred_at_t1 = pred_at_t0.item(), pred_at_t1.item()
    else:
        pred_at_t0 = model_h0.predict(test_data)[-1]
        pred_at_t1 = model_h1.predict(test_data)[-1]
    return pred_at_t0, pred_at_t1



def predict_dst(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float) -> Tuple[float, float]:
    """
    Take all of the data up until time t-1, and then make predictions for
    times t and t+1.
    Parameters
    ----------
    solar_wind_7d: pd.DataFrame
        The last 7 days of satellite data up until (t - 1) minutes [exclusive of t]
    satellite_positions_7d: pd.DataFrame
        The last 7 days of satellite position data up until the present time [inclusive of t]
    latest_sunspot_number: float
        The latest monthly sunspot number (SSN) to be available
    Returns
    -------
    predictions : Tuple[float, float]
        A tuple of two predictions, for (t and t + 1 hour) respectively; these should
        be between -2,000 and 500.
    """
    # make a copy so we dont modify the main dataframe
    solar_wind_7d = solar_wind_7d.copy()
    satellite_positions_7d = satellite_positions_7d.copy()
    satellite_positions_7d.reset_index(inplace=True)
    satellite_positions_7d.loc[:, 'period'] = 'test'

    # preprocess the solar wind features
    solar_wind_7d = solar_wind_preprocessing(solar_wind_7d,
                                             features=default.init_features)
    # preprcess the satellite position features
    satellite_positions_7d = stl_preprocessing(satellite_positions_7d)
    # calculate features solar wind features
    timestep = solar_wind_7d.index[-1].ceil('H')
    test_data = one_chunk_to_dataframe(solar_wind_7d, timestep)
    test_data['period'] = 'test'

    satellite_positions_7d.drop(['period', 'timedelta'], axis=1, inplace=True)
    satellite_positions_7d = satellite_positions_7d.tail(1)
    satellite_positions_7d.reset_index(drop=True, inplace=True)
    test_data = pd.concat((test_data, satellite_positions_7d), axis=1)

    # test_data = merge_daily(test_data, satellite_positions_7d)
    test_data["smoothed_ssn"] = np.log(latest_sunspot_number)
    # print('done')
    # Make a prediction
    # init the prediction at 0
    prediction_at_t0 = 0
    prediction_at_t1 = 0
    # for every experiment
    # print('start predicting')
    # print(test_data.shape)
    for experiment, experiment_repo in repo.items():
        # import the models
        # print(f'predicting using experiment {experiment}')
        model_h0 = experiment_repo['model_h0']
        model_h1 = experiment_repo['model_h1']
        pipeline = experiment_repo['pipeline']

        # test_data_e = test_data.copy()
        # print('applying preprocessing pipeline')
        test_data_e = pipeline.transform(test_data)
        features = sorted([feature for feature in test_data_e.columns
                           if feature not in default.ignore_features])

        pred_at_t0, pred_at_t1 = predict(test_data_e.loc[:, features],
                                         model_h0=model_h0,
                                         model_h1=model_h1)
        # predict and sum it to the total prediction
        prediction_at_t0 += pred_at_t0
        prediction_at_t1 += pred_at_t1
        # print('done')
    # divide by the number of experiments
    prediction_at_t0 /= len(repo)
    prediction_at_t1 /= len(repo)

    # Optional check for unexpected values
    if not np.isfinite(prediction_at_t0):
        prediction_at_t0 = -12
    if not np.isfinite(prediction_at_t1):
        prediction_at_t1 = -12

    return prediction_at_t0, prediction_at_t1


if __name__ == '__main__':
    import load_data
    import time
    data_path = Path('training_data/')
    dst_labels = load_data.read_csv(data_path / 'dst_labels.csv')
    solar_wind = load_data.read_feather(data_path / 'solar_wind.feather')
    sunspots = load_data.read_csv(data_path / 'sunspots.csv')
    stl_pos = load_data.read_csv(data_path / 'satellite_positions.csv')

    date = pd.to_timedelta(7, unit='d')
    # date = pd.to_timedelta('111 days 04:00:00')
    one_minute = pd.to_timedelta("1 minute")
    seven_days = pd.to_timedelta("7 days")
    solar_wind = solar_wind[solar_wind['period'] == 'train_a']
    sunspots = sunspots[sunspots['period'] == 'train_a']
    stl_pos = stl_pos[stl_pos['period'] == 'train_a']
    solar_wind.set_index(['timedelta'], inplace=True)
    stl_pos.set_index(['timedelta'], inplace=True)
    sunspots.set_index(['timedelta'], inplace=True)
    t_minus_7 = date - seven_days
    # last seven days of solar wind data except for the current minute
    solar_wind = solar_wind[t_minus_7: date - one_minute]
    stl_pos = (stl_pos.loc[:date].iloc[-7:, :])
    sunspots = sunspots.loc[:date]
    latest_sunspot_number = sunspots['smoothed_ssn'].values[-1]

    start = time.time()
    t0, t1 = predict_dst(solar_wind, stl_pos, latest_sunspot_number)
    end = time.time()
    print(dst_labels[dst_labels['timedelta'] == date])
    print(t0, t1, end-start)
