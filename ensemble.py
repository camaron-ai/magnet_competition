import pandas as pd
import numpy as np
from pathlib import Path
import load_data
import default
import logging
import click
import os
from typing import List
from metrics import compute_metrics
from itertools import combinations

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


def ensemble(data: pd.DataFrame,
             prediction: List[str] = default.yhat):
    ensemble_pred = data.groupby(['period', 'timedelta'])
    ensemble_pred = ensemble_pred[prediction].mean()
    ensemble_pred.reset_index(inplace=True)
    return ensemble_pred


def every_combination(combination, start: int = 2):
    output = []
    for r in range(start, len(combination) + 1):
        output += list(combinations(combination, r))
    return output


def main():
    path = Path('experiments')
    experiment_list = os.listdir(path)

    predictions = []
    for experiment in experiment_list:
        path_to_pred = path.joinpath(experiment,
                                     'prediction', 'valid.csv')
        if not os.path.exists(path_to_pred):
            continue
        pred_exp = load_data.read_csv(path_to_pred)
        pred_exp = pred_exp.assign(experiment=experiment)
        predictions.append(pred_exp)
    predictions = pd.concat(predictions)
    predictions.set_index('experiment', inplace=True)
    target = predictions.drop_duplicates(subset=['period',
                                         'timedelta'])
    target.reset_index(drop=True, inplace=True)
    target.drop(columns=default.yhat, inplace=True)

    results = []

    for combination in every_combination(predictions.index.unique()):
        predictions_combination = predictions.loc[list(combination)]
        predictions_ensemble = ensemble(predictions_combination)
        target_ensemble = target.merge(predictions_ensemble,
                                       on=['period', 'timedelta'],
                                       how='left')

        assert target_ensemble[default.yhat].isna().sum().sum() == 0
        combination_metric = compute_metrics(target_ensemble)
        combination_metric['experiment'] = '_'.join(combination)
        results.append(combination_metric)
        print(combination_metric)
    results = pd.DataFrame(results)
    results.sort_values(by='rmse', inplace=True)
    print(results.head(10))
    results.to_csv('ensemble.csv', index=False)


if __name__ == '__main__':
    main()
