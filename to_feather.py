from pathlib import Path
import load_data
import logging


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


def main():
    config = load_data.read_config_file('./config/config.yml')

    directories = config['directories']
    data_path = Path(directories['data'])
    logging.info('reading solar wind data..')
    solar_wind = load_data.read_csv(data_path / 'solar_wind.csv')
    logging.info('saving to feather..')
    solar_wind.to_feather(data_path / 'solar_wind.feather')


if __name__ == '__main__':
    main()
