from pathlib import Path
import load_data
import logging


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


def main():
    """
    This function will save the solar wind data
    as a Feather file
    """
    # read the main config file
    config = load_data.read_config_file('./config/config.yml')
    # get the path to the CSV File
    directories = config['directories']
    data_path = Path(directories['data'])
    logging.info('reading solar wind data..')
    # reading CSV file
    solar_wind = load_data.read_csv(data_path / 'solar_wind.csv')
    logging.info('saving to feather..')
    # saving as feather file
    solar_wind.to_feather(data_path / 'solar_wind.feather')


if __name__ == '__main__':
    main()
