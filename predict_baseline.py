import pandas as pd
import numpy as np
from preprocessing.base import preprocessing
import logging
from sklearn.linear_model import LinearRegression
import joblib
from typing import Tuple

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


init_features = ['bx_gse', 'by_gse', 'bz_gse', 'theta_gsm', 'bt',
                 'density', 'speed', 'temperature']
ignore_features = ['timedelta', 'period', 't0', 't1']

# loading models
linear_h0 = joblib.load('h0_baseline.pkl')
linear_h1 = joblib.load('h1_baseline.pkl')


def predict_dst(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
) -> Tuple[float, float]:
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

    test_data = preprocessing(solar_wind_7d, sunspots, features=init_features)
    features = [feature for feature in test_data.columns
                if feature not in ignore_features]
    # logging.info(f'modeling using {len(features)} features')
    # logging.info(f'{features[:30]}')

    # Make a prediction
    prediction_at_t0 = linear_h0.predict(test_data.loc[:, features])[-1]
    prediction_at_t1 = linear_h1.predict(test_data.loc[:, features])[-1]

    # Optional check for unexpected values
    if not np.isfinite(prediction_at_t0):
        prediction_at_t0 = -12
    if not np.isfinite(prediction_at_t1):
        prediction_at_t1 = -12

    return prediction_at_t0, prediction_at_t1
