from dplr.data import create_dl
from dplr import predict_dl
import numpy as np
import pandas as pd
from numpy.random import permutation
from joblib import Parallel, delayed


def permutation_importance(model, data,
                           features,
                           target,
                           score_func,
                           times: int = 10,
                           n_jobs=1):

    def _score(data):
        _, dl = create_dl(data, features=features)
        output = predict_dl(model, dl)
        prediction = output['prediction'].numpy()
        error = score_func(data[target], prediction)
        return error

    def permutated_score(feature, base_score):
        feature_scores = []
        for _ in range(times):
            permuted_data = data.copy()
            permuted_data[feature] = permutation(permuted_data[feature])
            feature_score = _score(permuted_data)
            feature_scores.append(feature_score)
            del permuted_data
        feature_score = np.mean(feature_scores)
        feature_score_std = np.std(feature_scores)
        feature_importance = {'feature': feature,
                              'score': feature_score,
                              'std': feature_score_std,
                              'importance': feature_score-base_score,
                              }
        return feature_importance
    base_score = _score(data)
    fi = Parallel(n_jobs=n_jobs)(delayed(permutated_score)(feature, base_score)
                                 for feature in features)
    fi = pd.DataFrame(fi)
    fi.sort_values(by='importance', inplace=True, ascending=False)
    fi.reset_index(drop=True, inplace=True)
    return fi
