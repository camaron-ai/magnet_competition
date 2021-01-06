import pandas as pd
import numpy as np


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


def compute_metrics(data: pd.DataFrame,
                    target='t',
                    yhat='yhat',
                    suffix=''):
    return {f'h0_rmse{suffix}': rmse(data[f'{target}0'], data[f'{yhat}_t0']),
            f'h1_rmse{suffix}': rmse(data[f'{target}1'], data[f'{yhat}_t1']),
            f'rmse{suffix}': rmse(data[[f'{target}0', f'{target}1']],
                                  data[[f'{yhat}_t0', f'{yhat}_t1']])
              }
