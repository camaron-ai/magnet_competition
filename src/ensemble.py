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
    """
    A function to ensemble preds by grouping by ['period', 'timedelta']
    # Parameters
    data: `pd.DataFrame`
        A dataframe containing the predictions
    prediction: `List[str]`, optional (defulat=['yhat_t0', 'yhat_t1'])
        the name of the prediction columns
    """
    ensemble_pred = data.groupby(['period', 'timedelta'])
    ensemble_pred = ensemble_pred[prediction].mean()
    ensemble_pred.reset_index(inplace=True)
    return ensemble_pred


def every_combination(combination: List[str], start: int = 2,
                      upto: int = None):
    """
    This function is use to compute every possible combination
    for a list of values.
    Be careful of using this function without specifyng the upto parameter,
    because the function will produce a output of the following lenght
    (n choose {start}) + (n choose {start} + 1) + ... + (n choose {len(combination) - 1})) + 1
    and depending of the lenght of the input this number can reach high values.

    # Parameters
    combination: `List[str]`
        A list of all possible values
    start: `int`, optional(default=2)
        minimum lenght of the subsequence
    upto: `int`, optional(defualt=None)
        maximum lenght of the subsequence

    # Returns
    List[Tuple[str]]: List of all possible combination

    # Example
    >>> list_of_values = ['hello', 'world', '!']
    >>> all_combination = every_combination(list_of_values)
    >>> all_combination
    [('hello', 'world'), ('world', '!'),
     ('hello', '!'), ('hello', 'world', '!')]
    >>> possible_2_combination = every_combination(list_of_values, upto=2)
    >>> possible_2_combination
    [('hello', 'world'), ('world', '!'), ('hello', '!')]
    """
    # if upto is not given
    # we will compute every possible combination up to len(combination)
    if upto is None:
        upto = len(combination)
    output = []
    # compute every combination
    for r in range(start, upto + 1):
        output += list(combinations(combination, r))
    return output


@click.command()
@click.option('--upto', default=None, type=int)
def main(upto):
    """
    A function to find which experiments we have to ensemble
    to get the best score.
    # Parameters
    upto: `int`, optional (defualt=None)
        maximum number of models to ensemble
        """
    # we assume all the experiments are saved
    # in the experiments folder
    path = Path('experiments')
    # get a list of all experiments name
    experiment_list = os.listdir(path)
    assert len(experiment_list) > 1, \
           'there is not enough experiments to ensemble'
    predictions = []
    # for every experiment
    for experiment in experiment_list:
        # create a path to the valid prediction file
        path_to_pred = path.joinpath(experiment,
                                     'prediction', 'valid.csv')
        if not os.path.exists(path_to_pred):
            continue
        # if this file exists, we read it and
        # set the experiment column to the name of this experiment
        pred_exp = load_data.read_csv(path_to_pred)
        pred_exp = pred_exp.assign(experiment=experiment)
        predictions.append(pred_exp)
    # concat all the predictions
    predictions = pd.concat(predictions)
    predictions.set_index('experiment', inplace=True)
    # create the target by dropping all duplicates
    target = predictions.drop_duplicates(subset=['period',
                                         'timedelta'])
    target.reset_index(drop=True, inplace=True)
    target.drop(columns=default.yhat, inplace=True)

    results = []
    # compute every experiment combination possible
    for combination in every_combination(predictions.index.unique(),
                                         upto=upto):
        # get the predictions for this specific combination
        predictions_combination = predictions.loc[list(combination)]
        # ensemble
        predictions_ensemble = ensemble(predictions_combination)
        # join the predictions to the target file
        target_ensemble = target.merge(predictions_ensemble,
                                       on=['period', 'timedelta'],
                                       how='left')
        # check there is non nan values
        assert target_ensemble[default.yhat].isna().sum().sum() == 0
        # compute the metrics
        combination_metric = compute_metrics(target_ensemble)
        combination_metric['experiment'] = '_'.join(combination)
        combination_metric['n_model'] = len(combination)
        results.append(combination_metric)
        print(combination_metric)
    results = pd.DataFrame(results)
    # sort by the rmse error
    results.sort_values(by='rmse', inplace=True)
    # print the top 10 models
    print(results.head(10))
    # save the ensemble results in a CSV file
    results.to_csv('ensemble.csv', index=False)


if __name__ == '__main__':
    main()
