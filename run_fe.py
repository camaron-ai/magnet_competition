from pathlib import Path
import load_data
from preprocessing.base import create_target, stl_preprocessing
from preprocessing.base import merge_daily, solar_wind_preprocessing
from preprocessing.base import split_into_period
import time
import logging
import click
import default
import warnings
import numpy as np

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


@click.command()
@click.option('--use_sample', type=click.BOOL, default=False)
def main(use_sample: bool = False):
    logging.info(f'use_sample={use_sample}')
    # use_sample=True
    logging.info('reading config file')
    config = load_data.read_config_file('./config/config.yml')
    # directories
    directories = config['directories']
    data_path = Path(directories['data'])

    # reading gt data
    solar_wind_file = ('sample_solar_wind.feather'
                       if use_sample else 'solar_wind.feather')
    logging.info('reading training data')
    dst_labels = load_data.read_csv(data_path / 'dst_labels.csv')
    solar_wind = load_data.read_feather(data_path / solar_wind_file)
    sunspots = load_data.read_csv(data_path / 'sunspots.csv')
    stl_pos = load_data.read_csv(data_path / 'satellite_positions.csv')

    logging.info('preprocessing solar wing')
    # preprocessing solar wind
    solar_wind.set_index('timedelta', inplace=True)
    solar_wind = solar_wind_preprocessing(solar_wind)
    logging.info('computing features')
    start = time.time()
    data = split_into_period(solar_wind,
                             features=default.init_features,
                             n_jobs=8)
    elapsed_time = (time.time()-start)/60
    logging.info(f'elapsed time {elapsed_time:.4f}')

    # create target
    logging.info('merging other datasets')
    target = create_target(dst_labels)
    stl_pos = stl_preprocessing(stl_pos)
    sunspots['smoothed_ssn'] = np.log(sunspots['smoothed_ssn'])
    data = merge_daily(data, stl_pos)
    data = merge_daily(data, sunspots)

    data = data.merge(target, how='left', on=['period', 'timedelta'])
    data.dropna(subset=['t0', 't1'], inplace=True)
    data.reset_index(inplace=True, drop=True)
    logging.info('saving')
    output_filename = 'fe' if not use_sample else 'fe_sample'
    data.to_feather(f'training_data/{output_filename}.feather')


if __name__ == '__main__':
    main()
