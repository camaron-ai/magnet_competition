import pandas as pd
from pathlib import Path
import load_data
import logging
import click

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)

default_frac = 0.4


@click.command()
@click.option('--frac', type=click.FLOAT, default=default_frac)
def main(frac: float = default_frac):
    logging.info(f'making sample of {frac}%')
    logging.info('reading config file')
    config = load_data.read_config_file('./config/config.yml')
    directories = config['directories']
    data_path = Path(directories['data'])
    # reading gt data
    logging.info('reading training data')
    solar_wind = load_data.read_csv(data_path / 'solar_wind.csv')

    logging.info('splitting dataset')
    _, valid_idx = load_data.split_train_data(solar_wind, test_frac=frac,
                                              eval_mode=True)

    sample_data = solar_wind.loc[valid_idx, :]
    sample_data.reset_index(drop=True, inplace=True)
    logging.info('saving file..')
    sample_data.to_csv(data_path / 'sample_solar_wind.csv', index=False)


if __name__ == '__main__':
    main()
