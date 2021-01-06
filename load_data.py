import yaml
import os
from typing import Union
import pandas as pd
import numpy as np
path_type = Union[str, os.PathLike]


def read_config_file(path_to_file: path_type):
    error_message = f'{ path_to_file } yaml file does not exists'
    assert os.path.exists(path_to_file), error_message
    with open(path_to_file, 'r') as file:
        return yaml.safe_load(file)


def read_csv(path_to_csv: path_type, **kargs) -> pd.DataFrame:
    return pd.read_csv(path_to_csv, parse_dates=['timedelta'],
                       date_parser=pd.to_timedelta, **kargs)


def split_train_data(data: pd.DataFrame, test_frac: float = 0.2,
                     eval_mode: bool = True):
    test_size = int(len(data) * test_frac)
    train_indexes = np.arange(len(data))
    if eval_mode:
        valid_indexes = data.groupby('period').tail(test_size // 3).index
        train_indexes = train_indexes[~np.isin(train_indexes, valid_indexes)]
        return train_indexes, valid_indexes
    else:
        valid_indexes = train_indexes[-test_size:]
        return train_indexes, valid_indexes
