import yaml
import os
from typing import Union
import pandas as pd
path_type = Union[str, os.PathLike]


def read_config_file(path_to_file: path_type):
    error_message = f'{ path_to_file } yaml file does not exists'
    assert os.path.exists(path_to_file), error_message
    with open(path_to_file, 'r') as file:
        return yaml.safe_load(file)


def read_csv(path_to_csv: path_type, **kargs) -> pd.DataFrame:
    return pd.read_csv(path_to_csv, parse_dates=['timedelta'],
                       date_parser=pd.to_timedelta, **kargs)

