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


def get_raw_importances(model):
    methods = ['feature_importances_', 'coef_']
    for method in methods:
        importances = getattr(model, method, None)
        if importances is not None:
            return importances
    return None


def feature_importances(model, features):
    importances = get_raw_importances(model)
    if importances is None:
        return None
    importances = np.abs(importances)
    fi = pd.DataFrame({'feature': features,
                       'importance': importances})
    fi.sort_values(by='importance', ascending=False, inplace=True)
    fi.reset_index(drop=True, inplace=True)
    return fi


def make_error_plot(prediction: pd.DataFrame):
    pass
