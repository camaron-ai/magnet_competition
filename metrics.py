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
