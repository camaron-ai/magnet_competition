import pandas as pd
import numpy as np
from pathlib import Path
import load_data
from preprocessing.base import preprocessing, create_target
import gc
import logging
import click
from sklearn.linear_model import LinearRegression
import joblib

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


init_features = ['bx_gse', 'by_gse', 'bz_gse', 'theta_gsm', 'bt',
                 'density', 'speed', 'temperature']
ignore_features = ['timedelta', 'period', 't0', 't1']


def to_numpy(function):
    def _inner(*arrays):
        new_arrays = [array.to_numpy()
                      if isinstance(array, pd.DataFrame)
                      else array
                      for array in arrays]
        return function(*new_arrays)
    return _inner


@to_numpy
def rmse(target, yhat):
    return np.sqrt(np.square(target - yhat).mean())


@click.command()
@click.option('--eval_mode', type=click.BOOL, default=True)
def main(eval_mode: bool = True):
    logging.info(f'running baseline on eval_mode={eval_mode}')
    logging.info('reading config file')
    config = load_data.read_config_file('./config/config.yml')
    directories = config['directories']
    data_path = Path(directories['data'])

    # reading gt data
    logging.info('reading training data')
    dst_labels = load_data.read_csv(data_path / 'dst_labels.csv')
    solar_wind = load_data.read_csv(data_path / 'solar_wind.csv')
    sunspots = load_data.read_csv(data_path / 'sunspots.csv')

    logging.info('applying base preprocessing')
    data = preprocessing(solar_wind, sunspots, features=init_features)
    target = create_target(dst_labels)
    assert len(data) == len(target), \
           f'lenght do not match {(len(data), len(target))}'
    data = data.merge(target, on=['period', 'timedelta'], how='left')
    features = [feature for feature in data.columns
                if feature not in ignore_features]
    logging.info(f'modeling using {len(features)} features')
    logging.info(f'{features[:30]}')

    del solar_wind, sunspots, dst_labels
    gc.collect()

    logging.info('splitting dataset')
    train_idx, valid_idx = load_data.split_train_data(data, test_frac=0.2,
                                                      eval_mode=eval_mode)

    train_data = data.loc[train_idx, :]
    valid_data = data.loc[valid_idx, :]

    logging.info('training horizon 0 model')
    # making model for horizon 0
    linear_h0 = LinearRegression(normalize=True)
    linear_h0.fit(train_data.loc[:, features], train_data.loc[:, 't0'])

    logging.info('training horizon 1 model')
    # making model for horizon 1
    linear_h1 = LinearRegression(normalize=True)
    linear_h1.fit(train_data.loc[:, features], train_data.loc[:, 't1'])

    logging.info('prediction h0 and h1 models')
    valid_data['yhat_t0'] = linear_h0.predict(valid_data.loc[:, features])
    valid_data['yhat_t1'] = linear_h1.predict(valid_data.loc[:, features])

    errors = {'h0_rmse': rmse(valid_data['t0'], valid_data['yhat_t0']),
              'h1_rmse': rmse(valid_data['t1'], valid_data['yhat_t1']),
              'rmse': rmse(valid_data[['t0', 't1']],
                           valid_data[['yhat_t0', 'yhat_t1']])
              }
    print(errors)
    if not eval_mode:
        joblib.dump(linear_h0, 'h0_baseline.pkl')
        joblib.dump(linear_h1, 'h1_baseline.pkl')


if __name__ == '__main__':
    main()
