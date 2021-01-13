import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import Tuple
from preprocessing.base import preprocessing
import logging
import joblib
import default
from models import library as model_library
from collections import defaultdict
import gc


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
        # load everything need it
        model_h0 = joblib.load(experiment_path / 'model_h0.pkl')
        model_h1 = joblib.load(experiment_path / 'model_h1.pkl')
        pipeline = joblib.load(experiment_path / 'pipeline.pkl')
        # save it into the experiment's dict
        repo[experiment]['model_h0'] = model_h0
        repo[experiment]['model_h1'] = model_h1
        repo[experiment]['pipeline'] = pipeline
    return repo


repo = load_models()


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
    # Re-format data to fit into our pipeline
    sunspots = pd.DataFrame(index=solar_wind_7d.index,
                            columns=["smoothed_ssn"])
    sunspots["smoothed_ssn"] = latest_sunspot_number

    sunspots.reset_index(inplace=True)
    solar_wind_7d.reset_index(inplace=True)
    solar_wind_7d.loc[:, 'period'] = 'test'
    sunspots.loc[:, 'period'] = 'test'

    test_data = preprocessing(solar_wind_7d, sunspots,
                              satellite_positions_7d,
                              features=default.init_features)
    # logging.info(f'modeling using {len(features)} features')
    # logging.info(f'{features[:30]}')

    # Make a prediction
    # init the prediction at 0
    prediction_at_t0 = 0
    prediction_at_t1 = 0
    # for every experiment
    for experiment, experiment_repo in repo.items():
        # import the models
        model_h0 = experiment_repo['model_h0']
        model_h1 = experiment_repo['model_h1']
        pipeline = experiment_repo['pipeline']

        test_data_e = test_data.copy()
        test_data_e = pipeline.transform(test_data_e)
        features = [feature for feature in test_data_e.columns
                    if feature not in default.ignore_features]
        # predict and sum it to the total prediction
        prediction_at_t0 += model_h0.predict(test_data_e.loc[:, features])[-1]
        prediction_at_t1 += model_h1.predict(test_data_e.loc[:, features])[-1]
        del features, test_data_e
        gc.collect()
    # divide by the number of experiments
    prediction_at_t0 /= len(repo)
    prediction_at_t1 /= len(repo)

    # Optional check for unexpected values
    if not np.isfinite(prediction_at_t0):
        prediction_at_t0 = -12
    if not np.isfinite(prediction_at_t1):
        prediction_at_t1 = -12

    return prediction_at_t0, prediction_at_t1
