from pathlib import Path
import numpy as np
import pandas as pd
import load_data
import logging
import click
import joblib
import mlflow
import default
import os
from metrics import compute_metrics, calculate_error_on_test
from metrics import compute_metrics_per_period
import metrics
from models import library as model_library
from pipelines import build_pipeline
from dplr import Learner, predict_dl
import torch
from torch import optim
from dplr.callback import ModelCheckpointCallBack
from dplr.callback.record import MetricRecorderCallBack, Recoder
from dplr.callback import ProgressBarCallBack
from dplr.data import DataLoader, Dataset, DataBunch
from dplr.interpretation import permutation_importance


torch.manual_seed(2021)
logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)
device = 'cpu'
target_name = ['t0', 't1']
batch_size = 512


@click.command()
@click.argument('experiment_path', type=click.Path(exists=True))
@click.option('--eval_mode', type=click.BOOL, default=True)
@click.option('--use_sample', type=click.BOOL, default=False)
@click.option('--test_frac', type=float, default=0.2)
@click.option('-m', '--message', type=str, default=None)
def main(experiment_path: str, eval_mode: bool = True,
         use_sample: bool = False, test_frac: float = 0.2,
         message: str = None):
    experiment = os.path.basename(experiment_path)
    logging.info(f'running {experiment}')
    logging.info(f'eval_mode={eval_mode}, use_sample={use_sample}')

    logging.info('reading config file')
    experiment_path = Path(experiment_path)
    config = load_data.read_config_file('./config/config.yml')
    experiment_config = load_data.read_config_file(experiment_path / 'config.yml')
    pipeline_config = load_data.read_config_file(experiment_config['pipeline'])
    model_config = load_data.read_config_file(experiment_config['model'])

    directories = config['directories']
    data_path = Path(directories['data'])
    prediction_path = experiment_path / 'prediction'
    prediction_path.mkdir(exist_ok=True, parents=True)
    model_path = experiment_path / 'models'

    # reading fe data
    filename = ('fe' if not use_sample else 'fe_sample')
    logging.info('reading training data')
    data = load_data.read_feather(data_path / f'{filename}.feather')

    logging.info('splitting dataset')
    train_idx, valid_idx = load_data.split_train_data(data,
                                                      test_frac=test_frac,
                                                      eval_mode=eval_mode)
    train_data = data.loc[train_idx, :]
    valid_data = data.loc[valid_idx, :]

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    # importing pipeline
    logging.info('building pipeline')
    pipeline = build_pipeline(pipeline_config)
    logging.info(f'{pipeline}')

    # fit pipeline
    logging.info('training pipeline')
    pipeline.fit(train_data)
    # transform
    logging.info('transforming datasets')
    train_data = pipeline.transform(train_data)
    valid_data = pipeline.transform(valid_data)

    use_sigmoid = experiment_config.pop('use_sigmoid', False)
    y_limit = ((train_data['t0'].agg(('max', 'min')) * 1.2).to_list()
               if use_sigmoid else None)

    features = sorted([feature for feature in train_data.columns
                       if feature not in default.ignore_features])
    in_features = len(features)
    logging.info(f'modeling using {len(features)} features')
    logging.info(f'{features[:30]}')

    # creating datasets
    train_ds = Dataset.from_dataframe(train_data, features=features,
                                      target=target_name, device=device)
    valid_ds = Dataset.from_dataframe(valid_data, features=features,
                                      target=target_name, device=device)

    train_dl = DataLoader(dataset=train_ds,
                          batch_size=batch_size,
                          shuffle=True)
    not_shuffle_train_dl = DataLoader(dataset=train_ds,
                                      batch_size=batch_size,
                                      shuffle=False)
    valid_dl = DataLoader(dataset=valid_ds,
                          batch_size=batch_size,
                          shuffle=False)
    bunch = DataBunch(train_dl, valid_dl)

    # importing and init model
    model_instance = model_library[model_config['instance']]

    model = model_instance(in_features=in_features,
                           out_features=len(target_name),
                           y_limit=y_limit,
                           **model_config['parameters']).to(device=device)
    # init optimizer
    optimizer = optim.Adam(model.parameters(),
                           **experiment_config['optimizer'])

    # creating learner instance
    logging.info('creating learner instance')
    cbs = [Recoder, MetricRecorderCallBack(metrics.torch_rmse),
           ModelCheckpointCallBack, ProgressBarCallBack]

    learner = Learner(model, optimizer, bunch, callbacks=cbs)

    logging.info('training model')
    # making model for horizon 0
    epochs = experiment_config.pop('epochs', 10)
    learner.fit(epochs, seed=2020)

    # avg the last 5 epochs weights
    top_models = np.arange(epochs)[-5:]
    learner.modelcheckpoint.load_averaged_model(top_models)

    logging.info('prediction h0 and h1 models')
    valid_output = predict_dl(learner.model, valid_dl)
    train_output = predict_dl(learner.model, not_shuffle_train_dl)
    valid_data[['yhat_t0', 'yhat_t1']] = valid_output['prediction'].numpy()
    train_data[['yhat_t0', 'yhat_t1']] = train_output['prediction'].numpy()

    # computing metrics
    train_error = compute_metrics(train_data, suffix='_train')
    valid_error = compute_metrics(valid_data, suffix='_valid')

    train_error_period = compute_metrics_per_period(train_data,
                                                    suffix='_train')
    valid_error_period = compute_metrics_per_period(valid_data,
                                                    suffix='_valid')

    logging.info('errors')
    logging.info(f'{train_error}')
    logging.info(f'{valid_error}')
    logging.info('period errors')
    logging.info(f'{train_error_period}')
    logging.info(f'{valid_error_period}')
    if eval_mode:
        with mlflow.start_run(run_name=experiment):
            # saving predictions
            train_prediction = train_data.loc[:, default.keep_columns]
            train_prediction.to_csv(prediction_path / 'train.csv', index=False)
            # saving training progress
            learner.metrics_table.to_csv(experiment_path / 'trn_progress.csv',
                                         index=False)
            # saving errors
            train_error_period.to_csv(experiment_path / 'train_erros.csv',
                                      index=False)
            valid_error_period.to_csv(experiment_path / 'valid_erros.csv',
                                      index=False)
            # valid_prediction = valid_data.loc[:, default.keep_columns]
            valid_data.to_csv(prediction_path / 'valid.csv', index=False)
            # saving feature importances if there is aviable
            fi = permutation_importance(model=learner.model,
                                        data=valid_data,
                                        features=features,
                                        target=target_name,
                                        score_func=metrics.rmse)
            fi.to_csv(experiment_path / 'fi_h0.csv', index=False)
            fi.to_csv(experiment_path / 'fi_h1.csv', index=False)
            # saving to mlflow
            # saving metrics
            mlflow.log_metrics(train_error)
            mlflow.log_metrics(valid_error)
            # saving model parameters
            mlflow.log_params(model_config['parameters'])
            mlflow.log_params(experiment_config['optimizer'])
            mlflow.log_params({'epochs': epochs,
                               'use_sigmoid': use_sigmoid})
            tags = {'use_sample': use_sample,
                    'model_instance': model_config['instance'],
                    'experiment': experiment}
            if message is not None:
                tags['message'] = message
            mlflow.set_tags(tags)
    if not eval_mode:
        # creating model path
        test_error = calculate_error_on_test(train_data)
        test_error = pd.DataFrame([test_error])
        test_error.to_csv(experiment_path / 'check_test_error.csv',
                          index=False)
        model_path.mkdir(exist_ok=True, parents=True)
        joblib.dump(learner.model, model_path / 'model_h0.pkl')
        joblib.dump(pipeline, model_path / 'pipeline.pkl')


if __name__ == '__main__':
    main()
